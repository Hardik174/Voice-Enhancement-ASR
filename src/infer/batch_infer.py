from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.audio_io import load_audio, save_audio, TARGET_SR
from src.model import load_model


def enhance_tensor(model, wav: torch.Tensor, dry: float = 0.01) -> torch.Tensor:
    with torch.inference_mode():
        estimate = model(wav[None])[0]
    return (1.0 - dry) * estimate + dry * wav


def enhance_file(in_path: str, out_path: str, model_name: str = "dns64", checkpoint: str | None = None,
                 device: str = "cpu", dry: float = 0.01) -> None:
    model = load_model(model_name, checkpoint, device)
    wav = load_audio(in_path, TARGET_SR).to(device)
    enhanced = enhance_tensor(model, wav, dry=dry)
    save_audio(out_path, enhanced, TARGET_SR)


def main() -> None:
    p = argparse.ArgumentParser(description="Enhance WAV/MP3 call audio with causal Demucs speech denoiser.")
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="dns64", choices=["dns48", "dns64", "master64"])
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry", type=float, default=0.01)
    args = p.parse_args()
    enhance_file(args.input, args.out, args.model, args.checkpoint, args.device, args.dry)


if __name__ == "__main__":
    main()
