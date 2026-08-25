from __future__ import annotations

"""
Stress test for the SraVaani ASR stage on CPU.

DELIBERATELY MORE CONSERVATIVE than src/eval/stress_test_denoiser.py. SraVaani is a
~430M-parameter model (~900MB of weights on disk; expect roughly that much resident in memory
once loaded, plus activation memory per call, plus Python/PyTorch/transformers overhead on top).
Loading several full instances at once to test "concurrency" the way the denoiser test does
would eat most of a 16GB laptop's memory on the model alone, before a single inference runs.

What this measures instead:
  1. sequential_latency_test (default): per-utterance latency (P50/P95) and memory drift for a
     SINGLE loaded instance across many consecutive calls. Answers "how long does one call take"
     and "does memory stay flat across calls" -- not "how many calls can run in parallel."
  2. capped_concurrency_test (opt-in, --mode concurrency): multi-worker test, HARD-CAPPED at 4
     workers -- each worker loads its own model instance, so this refuses to even start above 4.
     Guarded the same way as the denoiser test: a step won't start if the ResourceGuard floor is
     already breached, though the guard cannot undo memory a model load has already claimed.

Requires HF_TOKEN (SraVaani is gated) -- same as src/asr/transcribe.py. Uses a synthetic noise
WAV by default so you don't need a real recording on hand; pass --audio-file to test against a
real one (recommended once the synthetic run looks stable -- real speech content can behave
differently from silence/noise).

Usage (from the voice-enhancement project root, with the venv active):
    python -m src.eval.stress_test_asr --mode sequential --n-calls 10
    python -m src.eval.stress_test_asr --mode concurrency --max-workers 2
"""

import argparse
import json
import statistics
import struct
import time
import wave
from pathlib import Path
from typing import List, Optional

from src.asr.transcribe import AsrConfig, SraVaaniASR
from src.eval.resource_guard import ResourceGuard


def _write_synthetic_wav(path: str, seconds: float = 5.0, sample_rate: int = 16000) -> None:
    import random

    n_samples = int(seconds * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", random.randint(-500, 500)) for _ in range(n_samples))
        wf.writeframes(frames)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return s[idx]


def sequential_latency_test(
    n_calls: int = 10,
    audio_file: Optional[str] = None,
    asr_config: Optional[AsrConfig] = None,
    min_free_mb: int = 998,
    report_path: str = "reports/stress_asr_sequential.json",
) -> dict:
    guard = ResourceGuard(min_free_mb=min_free_mb)
    guard.start()

    tmp_audio = None
    if audio_file is None:
        Path("reports").mkdir(parents=True, exist_ok=True)
        tmp_audio = str(Path("reports") / "_stress_asr_synthetic.wav")
        _write_synthetic_wav(tmp_audio)
        audio_file = tmp_audio

    asr = SraVaaniASR(asr_config)
    latencies: List[float] = []
    samples = []
    load_seconds: Optional[float] = None
    try:
        # Load once, outside the timed loop, so model-load time doesn't pollute per-call latency.
        # Calling the "private" _load() directly (rather than transcribe_file, which also loads
        # lazily) is intentional here -- it's the only way to time load and inference separately.
        load_start = time.perf_counter()
        asr._load()
        load_seconds = time.perf_counter() - load_start
        print(f"[stress_asr] Model load took {load_seconds:.1f}s")

        for i in range(n_calls):
            if guard.should_abort:
                print(f"[stress_asr] Aborting at call {i}: memory floor hit.")
                break
            t0 = time.perf_counter()
            asr.transcribe_file(audio_file)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            snap = guard.snapshot()
            samples.append({"call": i, "elapsed_s": round(elapsed, 3), **snap})
            print(f"[stress_asr] call={i} elapsed={elapsed:.2f}s used_mb={snap['used_mb']}")
    finally:
        guard.stop()
        if tmp_audio:
            Path(tmp_audio).unlink(missing_ok=True)

    report = {
        "n_calls_completed": len(latencies),
        "n_calls_requested": n_calls,
        "load_seconds": round(load_seconds, 2) if load_seconds is not None else None,
        "p50_latency_s": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
        "aborted_on_memory": guard.should_abort,
        "peak_used_mb": round(guard.peak_used_mb, 1),
        "samples": samples,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[stress_asr] Report written to {report_path}")
    return report


def capped_concurrency_test(
    max_workers: int = 2,
    n_calls_per_worker: int = 3,
    audio_file: Optional[str] = None,
    asr_config: Optional[AsrConfig] = None,
    min_free_mb: int = 998,
    report_path: str = "reports/stress_asr_concurrency.json",
) -> dict:
    """Hard-capped at max_workers<=4. Each worker loads its OWN model instance -- that's
    ~900MB+ resident per worker before any inference runs. Do not raise --max-workers past what
    you have confirmed you can spare; the guard can refuse to start the NEXT step, but it can't
    undo memory a model load has already claimed mid-step."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers > 4:
        raise SystemExit(
            "Refusing max_workers > 4 for a ~900MB-per-instance model on a shared laptop. "
            "Lower --max-workers, or use --mode sequential instead."
        )

    guard = ResourceGuard(min_free_mb=min_free_mb)
    guard.start()

    tmp_audio = None
    if audio_file is None:
        Path("reports").mkdir(parents=True, exist_ok=True)
        tmp_audio = str(Path("reports") / "_stress_asr_synthetic.wav")
        _write_synthetic_wav(tmp_audio)
        audio_file = tmp_audio

    def worker() -> dict:
        asr = SraVaaniASR(asr_config)
        load_start = time.perf_counter()
        asr._load()
        load_seconds = time.perf_counter() - load_start
        times = []
        for _ in range(n_calls_per_worker):
            if guard.should_abort:
                break
            t0 = time.perf_counter()
            asr.transcribe_file(audio_file)
            times.append(time.perf_counter() - t0)
        return {"load_seconds": load_seconds, "inference_times": times}

    results = {}
    try:
        for n in [w for w in [1, 2, 4] if w <= max_workers]:
            if guard.should_abort:
                print(f"[stress_asr] Aborting before concurrency={n}: memory floor hit.")
                break
            print(f"[stress_asr] Running {n} concurrent ASR worker(s), each with its own model instance...")
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(worker) for _ in range(n)]
                all_load_times: List[float] = []
                all_inference_times: List[float] = []
                for f in as_completed(futures):
                    r = f.result()
                    all_load_times.append(r["load_seconds"])
                    all_inference_times.extend(r["inference_times"])
            snap = guard.snapshot()
            results[n] = {
                "mean_load_seconds": round(statistics.mean(all_load_times), 2) if all_load_times else None,
                "mean_inference_latency_s": round(statistics.mean(all_inference_times), 3) if all_inference_times else None,
                **snap,
            }
            print(
                f"[stress_asr]   concurrency={n} mean_load_seconds={results[n]['mean_load_seconds']} "
                f"mean_inference_latency_s={results[n]['mean_inference_latency_s']} used_mb={snap['used_mb']}"
            )
    finally:
        guard.stop()
        if tmp_audio:
            Path(tmp_audio).unlink(missing_ok=True)

    report = {
        "max_workers_tested": max_workers,
        "aborted_on_memory": guard.should_abort,
        "peak_used_mb": round(guard.peak_used_mb, 1),
        "steps": results,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[stress_asr] Report written to {report_path}")
    return report

def shared_instance_concurrency_test(
    max_workers: int = 8,
    n_calls_per_worker: int = 3,
    audio_file: Optional[str] = None,
    asr_config: Optional[AsrConfig] = None,
    min_free_mb: int = 1536,
    allow_unsafe_parallel: bool = False,
    report_path: str = "reports/stress_asr_shared_instance.json",
) -> dict:
    """The realistic production pattern: ONE loaded model instance, N threads dispatching calls
    to it -- not N separate instances (see capped_concurrency_test above). Pays the ~2GB model
    cost once, so it can safely test much higher concurrency on the same laptop.

    THREAD-SAFETY CAVEAT: SraVaani's decoding path (a custom trust_remote_code model) has not
    been verified thread-safe for truly concurrent calls on one shared instance -- autoregressive
    TDT-CTC decoding can carry internal state/buffers that a naive multi-thread call could race
    on, which could silently produce a corrupted transcript rather than crash loudly. By DEFAULT
    this test serializes actual inference calls behind a lock (threads still queue realistically,
    only one call runs the model at a time) -- that still safely measures memory behavior and
    queueing throughput. Pass allow_unsafe_parallel=True / --allow-unsafe-parallel only if you
    have separately confirmed the model is safe for concurrent calls, and don't trust the output
    blindly even then.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    guard = ResourceGuard(min_free_mb=min_free_mb)
    guard.start()

    tmp_audio = None
    if audio_file is None:
        Path("reports").mkdir(parents=True, exist_ok=True)
        tmp_audio = str(Path("reports") / "_stress_asr_synthetic.wav")
        _write_synthetic_wav(tmp_audio)
        audio_file = tmp_audio

    asr = SraVaaniASR(asr_config)
    load_start = time.perf_counter()
    asr._load()
    load_seconds = time.perf_counter() - load_start
    print(f"[stress_asr] Shared instance loaded once in {load_seconds:.1f}s")

    call_lock = threading.Lock() if not allow_unsafe_parallel else None

    def call_once() -> float:
        t0 = time.perf_counter()
        if call_lock:
            with call_lock:
                asr.transcribe_file(audio_file)
        else:
            asr.transcribe_file(audio_file)
        return time.perf_counter() - t0

    results = {}
    try:
        for n in [w for w in [1, 2, 4, 8] if w <= max_workers]:
            if guard.should_abort:
                print(f"[stress_asr] Aborting before concurrency={n}: memory floor hit.")
                break
            print(
                f"[stress_asr] Shared instance, {n} concurrent caller(s) "
                f"({'SERIALIZED via lock' if call_lock else 'UNSAFE parallel'})..."
            )
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(call_once) for _ in range(n * n_calls_per_worker)]
                times = [f.result() for f in as_completed(futures)]
            wall = time.perf_counter() - t0
            snap = guard.snapshot()
            results[n] = {
                "wall_seconds": round(wall, 2),
                "mean_call_latency_s": round(statistics.mean(times), 3),
                "throughput_calls_per_sec": round(len(times) / wall, 3),
                **snap,
            }
            print(
                f"[stress_asr]   concurrency={n} throughput={results[n]['throughput_calls_per_sec']}/s "
                f"used_mb={snap['used_mb']}"
            )
    finally:
        guard.stop()
        if tmp_audio:
            Path(tmp_audio).unlink(missing_ok=True)

    report = {
        "mode": "shared_instance",
        "serialized": call_lock is not None,
        "load_seconds": round(load_seconds, 2),
        "aborted_on_memory": guard.should_abort,
        "peak_used_mb": round(guard.peak_used_mb, 1),
        "steps": results,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[stress_asr] Report written to {report_path}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="CPU stress test for the SraVaani ASR stage.")
    p.add_argument("--mode", choices=["sequential", "concurrency", "shared"], default="sequential")
    p.add_argument("--n-calls", type=int, default=10)
    p.add_argument("--audio-file", default=None, help="Real audio file; default generates synthetic noise")
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--n-calls-per-worker", type=int, default=3)
    # p.add_argument("--min-free-mb", type=int, default=998)
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--asr-repo-id", default="ARTPARK-IISc/SraVaani-1.0")
    p.add_argument("--asr-device", default=None)
    p.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN env var if set")
    p.add_argument("--min-free-mb", type=int, default=998,
                    help="Abort at the next checkpoint if available memory drops below this. "
                         "Keep this at 1536+ on Windows -- below ~1GB free, Windows itself can "
                         "start paging aggressively and the machine can feel frozen even before "
                         "anything technically crashes.")
    p.add_argument("--allow-unsafe-parallel", action="store_true",
                    help="--mode shared only: skip the safety lock and call the shared model "
                         "instance truly concurrently from multiple threads. Only use this if "
                         "you've separately verified SraVaani's decoding path is thread-safe.")
    args = p.parse_args()

    config = AsrConfig(repo_id=args.asr_repo_id, device=args.asr_device, hf_token=args.hf_token)

    if args.mode == "sequential":
        sequential_latency_test(
            args.n_calls, args.audio_file, config, args.min_free_mb,
            str(Path(args.report_dir) / "stress_asr_sequential.json"),
        )
    elif args.mode == "concurrency":
        capped_concurrency_test(
            args.max_workers, args.n_calls_per_worker, args.audio_file, config,
            max(args.min_free_mb, 998),
            str(Path(args.report_dir) / "stress_asr_concurrency.json"),
        )
    elif args.mode == "shared":
        shared_instance_concurrency_test(
            args.max_workers,
            args.n_calls_per_worker,
            args.audio_file,
            config,
            args.min_free_mb,
            args.allow_unsafe_parallel,
            str(Path(args.report_dir) / "stress_asr_shared_instance.json"),
        )


if __name__ == "__main__":
    main()
