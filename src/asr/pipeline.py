from __future__ import annotations

import argparse
import pathlib
import sys

# Torchaudio backend compatibility patch for DeepFilterNet
import types
import torchaudio
if not hasattr(torchaudio, "backend"):
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common

# Cross-platform compatibility fix for loading Linux-saved model hparams in omegaconf
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

import subprocess
from pathlib import Path
from typing import Optional

import torch

from src.audio_io import load_audio, save_audio, TARGET_SR
from src.model import load_model
from src.asr.transcribe import AsrConfig, transcribe_dir_to_jsonl, AUDIO_EXTENSIONS
from df.enhance import init_df, enhance


def enhance_dir(
    raw_dir: str,
    enhanced_dir: str,
    model_name: str = "dns64",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    dry: float = 0.01,
    run_deepfilter: bool = False,
    prefix: Optional[str] = None,
) -> None:
    """Enhance every audio file in raw_dir, writing same-named files to enhanced_dir.

    Loads the denoiser once and reuses it across files -- unlike calling
    src.infer.batch_infer.enhance_file() in a loop, which reloads the model from disk on every
    single call. For a handful of files that's negligible; for a real call archive it is not.
    """
    raw_dir_p, enhanced_dir_p = Path(raw_dir), Path(enhanced_dir)
    enhanced_dir_p.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in raw_dir_p.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {raw_dir}")

    model = load_model(model_name, checkpoint, device)
    if run_deepfilter:
        df_model, df_state, _ = init_df()

    for p in files:
        wav = load_audio(str(p), TARGET_SR).to(device)
        with torch.inference_mode():
            estimate = model(wav[None])[0]
        enhanced = (1.0 - dry) * estimate + dry * wav

        sr = TARGET_SR
        if run_deepfilter:
            # Resample denoised (16kHz) to 48kHz
            resample_48k = torchaudio.transforms.Resample(orig_freq=TARGET_SR, new_freq=48000).to(device)
            denoised_48k = resample_48k(enhanced)
            
            # DeepFilterNet enhance function expects CPU tensor
            denoised_cpu = denoised_48k.cpu()
            enhanced_48k_cpu = enhance(df_model, df_state, denoised_cpu)
            
            enhanced = enhanced_48k_cpu.to(device)
            sr = 48000

        out_name = p.with_suffix(".wav").name
        if prefix:
            out_name = f"{prefix}{out_name}"
        save_audio(str(enhanced_dir_p / out_name), enhanced, sr)


def run_wer_delta(raw_jsonl: str, enhanced_jsonl: str, references: str) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "src.eval.compute_wer_delta",
            "--raw-transcripts", raw_jsonl,
            "--enhanced-transcripts", enhanced_jsonl,
            "--references", references,
        ],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Raw call audio -> denoiser enhancement -> SraVaani ASR -> WER-delta report."
    )
    p.add_argument("--raw-dir", required=True, help="Directory of raw (noisy) call audio")
    p.add_argument("--enhanced-dir", required=True, help="Where enhanced audio is written/read")
    p.add_argument("--references", default=None, help="JSONL: {id,text} ground-truth transcripts")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--denoiser-model", default="dns64", choices=["dns48", "dns64", "master64"])
    p.add_argument("--denoiser-checkpoint", default=None, help="Fine-tuned checkpoint, e.g. dns64_indic_v1.th")
    p.add_argument("--denoiser-device", default="cpu")
    p.add_argument("--dry", type=float, default=0.01)
    p.add_argument("--skip-enhance", action="store_true", help="enhanced-dir is already populated")
    p.add_argument("--deepfilter", action="store_true", help="Run DeepFilterNet 3 enhancement stage")
    p.add_argument("--asr-repo-id", default="ARTPARK-IISc/SraVaani-1.0")
    p.add_argument("--asr-device", default=None)
    p.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN env var if set")
    p.add_argument("--prefix", default=None, help="Prefix to add to enhanced audio file names")
    args = p.parse_args()

    if not args.skip_enhance:
        enhance_dir(
            args.raw_dir, args.enhanced_dir, args.denoiser_model, args.denoiser_checkpoint,
            args.denoiser_device, args.dry,
            run_deepfilter=args.deepfilter,
            prefix=args.prefix,
        )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = report_dir / "raw.jsonl"
    enhanced_jsonl = report_dir / "enhanced.jsonl"

    asr_config = AsrConfig(repo_id=args.asr_repo_id, device=args.asr_device, hf_token=args.hf_token)
    transcribe_dir_to_jsonl(args.raw_dir, str(raw_jsonl), config=asr_config)
    transcribe_dir_to_jsonl(args.enhanced_dir, str(enhanced_jsonl), config=asr_config, prefix_to_strip=args.prefix)

    if args.references:
        run_wer_delta(str(raw_jsonl), str(enhanced_jsonl), args.references)
    else:
        print("Skipping WER delta calculation: no references file provided.")


if __name__ == "__main__":
    main()
