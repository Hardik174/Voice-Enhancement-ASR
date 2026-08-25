from __future__ import annotations

"""
Concurrent + soak stress test for the causal-Demucs denoiser on CPU.

Answers two things the earlier single-stream benchmark (src/eval/benchmark_stream.py) doesn't:
  1. Concurrent-call RTF: what happens to real-time-factor as you add more simultaneous "calls"
     on this machine, not just one stream in isolation. This is the concurrent-load benchmark
     called out as missing in the earlier code review.
  2. Soak stability: does memory grow unboundedly over a long-running session (which would
     indicate a buffering bug, e.g. the AudioStreamBridge chunking issue fixed earlier), or does
     it plateau.

SAFE BY DESIGN FOR A LAPTOP WITH LOW MEMORY HEADROOM:
  - Ramps concurrency up one step at a time (1, 2, 4, 6, 8, 12, 16, capped at --max-concurrency)
    instead of jumping straight to a high number.
  - A ResourceGuard background thread (src/eval/resource_guard.py) signals abort at the next
    checkpoint if available memory drops below --min-free-mb (default 1536 MB) -- checked
    between every step, not just at the end.
  - Each simulated "call" pins itself to torch.set_num_threads(1) so N concurrent calls don't
    each try to grab every core -- this also matches how a real multi-call production
    deployment behaves (one thread per call).
  - Uses synthetic random audio, so no dependency on having real call recordings on hand.

Usage (from the voice-enhancement project root, with the venv active):
    python -m src.eval.stress_test_denoiser --mode concurrency --max-concurrency 4
    python -m src.eval.stress_test_denoiser --mode soak --soak-minutes 5
    python -m src.eval.stress_test_denoiser --mode both

Recommended first run on a memory-constrained machine: start with a small --max-concurrency
(e.g. 4) rather than the auto-detected (cpu_count - 1) default, confirm the report looks sane
and nothing got close to the memory floor, then raise it if you want a higher ceiling reading.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import torch

from src.model import load_model
from src.eval.resource_guard import ResourceGuard


def _synthetic_chunk(sample_rate: int, seconds: float) -> torch.Tensor:
    return torch.randn(1, int(sample_rate * seconds)) * 0.05  # quiet synthetic noise, mono


def _run_one_stream(model, sample_rate: int, seconds: float) -> float:
    """One simulated call: enhance `seconds` worth of audio, return RTF (compute_time / audio_duration)."""
    torch.set_num_threads(1)
    wav = _synthetic_chunk(sample_rate, seconds)
    start = time.perf_counter()
    with torch.inference_mode():
        model(wav[None])
    elapsed = time.perf_counter() - start
    return elapsed / seconds


def concurrency_ramp(
    model_name: str = "dns64",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    sample_rate: int = 16000,
    chunk_seconds: float = 4.0,
    max_concurrency: Optional[int] = None,
    min_free_mb: int = 1536,
    rtf_fail_threshold: float = 1.0,
    report_path: str = "reports/stress_denoiser_concurrency.json",
) -> dict:
    max_concurrency = max_concurrency or max(1, (os.cpu_count() or 2) - 1)
    model = load_model(model_name, checkpoint, device)
    model.eval()

    guard = ResourceGuard(min_free_mb=min_free_mb)
    guard.start()

    results = []
    try:
        steps = [c for c in [1, 2, 4, 6, 8, 12, 16] if c <= max_concurrency] or [1]
        for n in steps:
            if guard.should_abort:
                print(f"[stress_test] Aborting ramp before concurrency={n}: memory floor hit.")
                break

            print(f"[stress_test] Running concurrency={n} ...")
            rtfs: List[float] = []
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(_run_one_stream, model, sample_rate, chunk_seconds) for _ in range(n)]
                for f in as_completed(futures):
                    rtfs.append(f.result())
            wall = time.perf_counter() - t0

            snap = guard.snapshot()
            step_result = {
                "concurrency": n,
                "mean_rtf": round(statistics.mean(rtfs), 4),
                "max_rtf": round(max(rtfs), 4),
                "wall_seconds": round(wall, 3),
                **snap,
            }
            results.append(step_result)
            print(
                f"[stress_test]   mean_rtf={step_result['mean_rtf']} max_rtf={step_result['max_rtf']} "
                f"available_mb={snap['available_mb']}"
            )

            if step_result["max_rtf"] > rtf_fail_threshold:
                print(
                    f"[stress_test] Stopping ramp: max_rtf {step_result['max_rtf']} exceeded "
                    f"real-time threshold {rtf_fail_threshold} at concurrency={n}. This is the "
                    f"practical concurrent-call ceiling on this machine, not a crash."
                )
                break
    finally:
        guard.stop()

    report = {
        "model": model_name,
        "device": device,
        "chunk_seconds": chunk_seconds,
        "min_free_mb_floor": min_free_mb,
        "aborted_on_memory": guard.should_abort,
        "peak_used_mb": round(guard.peak_used_mb, 1),
        "min_available_seen_mb": (
            round(guard.min_available_seen_mb, 1) if guard.min_available_seen_mb != float("inf") else None
        ),
        "steps": results,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[stress_test] Report written to {report_path}")
    return report


def soak_test(
    model_name: str = "dns64",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    sample_rate: int = 16000,
    chunk_seconds: float = 1.0,
    soak_minutes: float = 10.0,
    min_free_mb: int = 1536,
    report_path: str = "reports/stress_denoiser_soak.json",
) -> dict:
    """Repeatedly enhance small chunks for soak_minutes, sampling memory every iteration.
    Looks for unbounded memory growth over time (a leak / buffering bug), not just a crash."""
    model = load_model(model_name, checkpoint, device)
    model.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    guard = ResourceGuard(min_free_mb=min_free_mb)
    guard.start()

    samples = []
    end_time = time.perf_counter() + soak_minutes * 60
    iteration = 0
    try:
        while time.perf_counter() < end_time and not guard.should_abort:
            wav = _synthetic_chunk(sample_rate, chunk_seconds)
            t0 = time.perf_counter()
            with torch.inference_mode():
                model(wav[None])
            elapsed = time.perf_counter() - t0
            snap = guard.snapshot()
            samples.append({"iteration": iteration, "elapsed_s": round(elapsed, 4), **snap})
            iteration += 1
            if iteration % 30 == 0:
                print(f"[soak_test] iter={iteration} used_mb={snap['used_mb']} available_mb={snap['available_mb']}")
    finally:
        guard.stop()

    used = [s["used_mb"] for s in samples]
    report = {
        "model": model_name,
        "iterations": iteration,
        "soak_minutes_requested": soak_minutes,
        "aborted_on_memory": guard.should_abort,
        "used_mb_start": used[0] if used else None,
        "used_mb_end": used[-1] if used else None,
        "used_mb_drift": round(used[-1] - used[0], 1) if len(used) > 1 else None,
        "peak_used_mb": round(guard.peak_used_mb, 1),
        "samples_tail": samples[-10:],
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[soak_test] Report written to {report_path}. used_mb drift={report['used_mb_drift']}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="CPU stress test for the causal-Demucs denoiser.")
    p.add_argument("--mode", choices=["concurrency", "soak", "both"], default="concurrency")
    p.add_argument("--model", default="dns64", choices=["dns48", "dns64", "master64"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-seconds", type=float, default=4.0)
    p.add_argument("--max-concurrency", type=int, default=None,
                    help="Default: (CPU cores - 1) on this machine. Start smaller on a "
                         "memory-constrained laptop, e.g. --max-concurrency 4.")
    p.add_argument("--min-free-mb", type=int, default=1536,
                    help="Abort the run at the next checkpoint if available system memory drops below this")
    p.add_argument("--rtf-fail-threshold", type=float, default=1.0)
    p.add_argument("--soak-minutes", type=float, default=10.0)
    p.add_argument("--report-dir", default="reports")
    args = p.parse_args()

    if args.mode in ("concurrency", "both"):
        concurrency_ramp(
            args.model, args.checkpoint, args.device, chunk_seconds=args.chunk_seconds,
            max_concurrency=args.max_concurrency, min_free_mb=args.min_free_mb,
            rtf_fail_threshold=args.rtf_fail_threshold,
            report_path=str(Path(args.report_dir) / "stress_denoiser_concurrency.json"),
        )
    if args.mode in ("soak", "both"):
        soak_test(
            args.model, args.checkpoint, args.device, soak_minutes=args.soak_minutes,
            min_free_mb=args.min_free_mb,
            report_path=str(Path(args.report_dir) / "stress_denoiser_soak.json"),
        )


if __name__ == "__main__":
    main()
