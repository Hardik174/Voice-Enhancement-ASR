from __future__ import annotations

"""
Run once, on one file: raw audio -> denoiser enhancement -> SraVaani transcript.

This is deliberately the single-file counterpart to src.asr.pipeline (which is directory-based,
requires a references.jsonl, and computes a WER-delta report across many files for evaluation).
Use this module for "I have one call recording, give me its transcript" -- no directories, no
references, no scoring, just one input file to one transcript.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from src.audio_io import load_audio, save_audio, TARGET_SR
from src.model import load_model
from src.asr.transcribe import AsrConfig, SraVaaniASR


def enhance_single_file(
    raw_path: str,
    enhanced_path: str,
    model_name: str = "dns64",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    dry: float = 0.01,
) -> None:
    """Enhance exactly one file. For many files, prefer src.asr.pipeline.enhance_dir, which
    loads the denoiser once and reuses it -- this function loads it fresh every call, which is
    correct (there's nothing to reuse) but not something you want in a loop."""
    model = load_model(model_name, checkpoint, device)
    wav = load_audio(raw_path, TARGET_SR).to(device)
    with torch.inference_mode():
        estimate = model(wav[None])[0]
    enhanced = (1.0 - dry) * estimate + dry * wav
    save_audio(enhanced_path, enhanced, TARGET_SR)


def run_once(
    raw_path: str,
    enhanced_path: str,
    asr_config: Optional[AsrConfig] = None,
    denoiser_model: str = "dns64",
    denoiser_checkpoint: Optional[str] = None,
    denoiser_device: str = "cpu",
    dry: float = 0.01,
    skip_enhance: bool = False,
    transcript_out: Optional[str] = None,
) -> dict:
    """Returns {"id","text"}. If skip_enhance is True, raw_path is transcribed directly instead
    of enhanced_path (useful for an A/B check against the same SraVaani call)."""
    if not skip_enhance:
        enhance_single_file(raw_path, enhanced_path, denoiser_model, denoiser_checkpoint, denoiser_device, dry)
        audio_for_asr = enhanced_path
    else:
        audio_for_asr = raw_path

    asr = SraVaaniASR(asr_config)
    text = asr.transcribe_file(audio_for_asr)
    result = {"id": Path(audio_for_asr).stem, "text": text}

    if transcript_out:
        out_p = Path(transcript_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="One command, one file: raw audio -> enhance -> SraVaani transcript."
    )
    p.add_argument("--raw", required=True, help="Path to the raw (noisy) input audio file")
    p.add_argument("--enhanced-out", default=None,
                    help="Where to write the enhanced audio. Defaults to <raw>_enhanced.wav next to --raw.")
    p.add_argument("--transcript-out", default=None, help="Optional path to save the transcript as JSON")
    p.add_argument("--skip-enhance", action="store_true", help="Transcribe --raw directly, skip denoising")
    p.add_argument("--denoiser-model", default="dns64", choices=["dns48", "dns64", "master64"])
    p.add_argument("--denoiser-checkpoint", default=None, help="Fine-tuned checkpoint, e.g. dns64_indic_v1.th")
    p.add_argument("--denoiser-device", default="cpu")
    p.add_argument("--dry", type=float, default=0.01)
    p.add_argument("--asr-repo-id", default="ARTPARK-IISc/SraVaani-1.0")
    p.add_argument("--asr-device", default=None)
    p.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN env var if set")
    args = p.parse_args()

    raw_path = Path(args.raw)
    enhanced_out = Path(args.enhanced_out) if args.enhanced_out else raw_path.with_name(
        raw_path.stem + "_enhanced" + raw_path.suffix
    )

    asr_config = AsrConfig(repo_id=args.asr_repo_id, device=args.asr_device, hf_token=args.hf_token)
    result = run_once(
        str(raw_path), str(enhanced_out), asr_config=asr_config,
        denoiser_model=args.denoiser_model, denoiser_checkpoint=args.denoiser_checkpoint,
        denoiser_device=args.denoiser_device, dry=args.dry,
        skip_enhance=args.skip_enhance, transcript_out=args.transcript_out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
