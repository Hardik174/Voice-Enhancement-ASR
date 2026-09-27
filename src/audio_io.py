from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16_000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load WAV/MP3/etc as mono float32 [1, T] at target_sr."""
    path = Path(path)
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Could not read audio. Install ffmpeg and ensure it is on PATH.")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(target_sr), str(tmp_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            data, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
            wav = torch.from_numpy(np.asarray(data)).float().reshape(1, -1)
            return wav if sr == target_sr else torchaudio.functional.resample(wav, sr, target_sr)
        finally:
            tmp_path.unlink(missing_ok=True)

    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def save_audio(path: str | Path, wav: torch.Tensor, sr: int = TARGET_SR) -> None:
    path = Path(path)
    if path.suffix.lower() not in (".wav", ".flac"):
        path = path.with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu().reshape(-1).clamp(-1, 1)
    sf.write(str(path), wav.numpy(), sr, subtype="PCM_16")
