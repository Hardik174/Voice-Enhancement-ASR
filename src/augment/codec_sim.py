from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
import soundfile as sf


def ffmpeg_codec(wav: torch.Tensor, codec: str, sr: int = 16000) -> torch.Tensor:
    """Optional codec simulation. Requires ffmpeg; falls back to input for codec=none."""
    if codec == "none":
        return wav.clone()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for codec simulation")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inp = d / "in.wav"
        sf.write(inp, wav.flatten().cpu().numpy(), sr, subtype="PCM_16")

        cmd = None
        if codec == "gsm_fr":
            # GSM full-rate is supported directly inside a WAV container.
            encoded = d / "out.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(inp),
                "-ar", "8000", "-ac", "1", "-acodec", "gsm", str(encoded)
            ]
        elif codec == "amr_nb":
            # AMR-NB cannot be muxed into WAV. Encode to an AMR container,
            # then decode it back to WAV for the rest of the pipeline.
            encoded = d / "out.amr"
            decoded = d / "decoded.wav"
            encode_cmd = [
                "ffmpeg", "-y", "-i", str(inp),
                "-ar", "8000", "-ac", "1",
                "-acodec", "libopencore_amrnb", "-b:a", "12.2k",
                "-f", "amr", str(encoded)
            ]
            subprocess.run(
                encode_cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            decode_cmd = [
                "ffmpeg", "-y", "-i", str(encoded),
                "-ar", "8000", "-ac", "1", "-acodec", "pcm_s16le",
                str(decoded)
            ]
            subprocess.run(
                decode_cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            encoded = decoded
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        if cmd is not None:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )

        data, out_sr = sf.read(encoded, dtype="float32")
        y = torch.from_numpy(data).reshape(1, -1)
        if out_sr != sr:
            import torchaudio
            y = torchaudio.functional.resample(y, out_sr, sr)
        return y
