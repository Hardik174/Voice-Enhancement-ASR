from __future__ import annotations

"""
ASR (text-generation) stage: ARTPARK-IISc/SraVaani-1.0
https://huggingface.co/ARTPARK-IISc/SraVaani-1.0

SraVaani is a ~430M-parameter FastConformer model with a hybrid TDT-CTC decoder, fine-tuned
for 65 Indian languages/dialects. It is a SEPARATE model from the causal-Demucs denoiser
(src/model.py) -- this module is the transcription stage that runs on the denoiser's output,
not a replacement for it.

READ BEFORE USING IN PRODUCTION:

1. GATED MODEL. You must accept the repo's access conditions on huggingface.co and provide an
   HF access token (env var HF_TOKEN by default, or --hf-token / AsrConfig.hf_token). Without
   this, model loading raises a clear RuntimeError rather than a cryptic HF 401.

2. trust_remote_code=True. Loading executes Python code shipped inside the model repo, not just
   weights. Review that code (the "Files" tab on the model page) before running it against
   production infra, the same as any other trust_remote_code model. This wrapper does not (and
   cannot) sandbox that code for you.

3. OFFLINE/BATCH, NOT LOW-LATENCY STREAMING. `model.transcribe(...)` processes a whole utterance
   at once. It gives you a text-generation stage for recorded/enhanced call audio (WER-delta
   eval, FW reprocessing, LC post-call transcripts) -- it does NOT give you incremental partial
   transcripts inside a live call. True streaming ASR needs NeMo's cache-aware streaming
   FastConformer decoding, which is a materially larger integration than wrapping this API.
   `StreamingSraVaaniASR` below is an explicit, clearly-labeled approximation (chunk + batch-
   transcribe each chunk) for prototyping only -- it is NOT real streaming and will cut words at
   chunk boundaries. Do not use it as a substitute for a real streaming ASR integration.

4. WER varies enormously by language for this model (see the model card's per-language table --
   e.g. Hindi ~12% vs. many regional dialects at 50-78%). Scope your evaluation to the languages
   you actually need before trusting aggregate numbers.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_REPO_ID = "ARTPARK-IISc/SraVaani-1.0"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")


@dataclass
class AsrConfig:
    repo_id: str = DEFAULT_REPO_ID
    device: Optional[str] = None  # resolved lazily so importing this module doesn't need torch+cuda probing
    hf_token: Optional[str] = None  # falls back to hf_token_env if not set
    hf_token_env: str = "HF_TOKEN"

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolve_token(self) -> str:
        token = self.hf_token or os.environ.get(self.hf_token_env)
        if not token:
            raise RuntimeError(
                f"No Hugging Face token found. Set {self.hf_token_env} (or pass hf_token=...) to "
                f"a token with access to {self.repo_id}. This model is gated: you must first "
                f"accept its access conditions at https://huggingface.co/{self.repo_id}."
            )
        return token


class SraVaaniASR:
    """Thin wrapper around SraVaani's AutoModel.transcribe() API. Lazy-loads the model on first
    use so importing this module (e.g. for tests) never triggers a download or requires a token."""

    def __init__(self, config: Optional[AsrConfig] = None):
        self.config = config or AsrConfig()
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers first: pip install transformers"
            ) from exc

        token = self.config.resolve_token()
        device = self.config.resolve_device()
        self._model = (
            AutoModel.from_pretrained(self.config.repo_id, trust_remote_code=True, token=token)
            .to(device)
            .eval()
        )

    def transcribe_files(self, paths: Sequence[str]) -> List[str]:
        """Batch-transcribe a list of audio file paths, preserving order."""
        self._load()
        paths = list(paths)
        if not paths:
            return []
        hyps = self._model.transcribe(paths, return_hypotheses=True)
        return [h.text for h in hyps]

    def transcribe_file(self, path: str) -> str:
        return self.transcribe_files([path])[0]


class StreamingSraVaaniASR:
    """NOT real streaming ASR -- see module docstring point 3. Splits audio into fixed-length
    chunks and batch-transcribes each one independently, which will cut words at chunk
    boundaries and gives no true low-latency partial output. Provided only so the pipeline has
    an explicit, honestly-labeled placeholder to replace with a real NeMo cache-aware streaming
    decoder rather than silently pretending the offline API is a streaming one."""

    def __init__(self, config: Optional[AsrConfig] = None, chunk_seconds: float = 8.0):
        self._asr = SraVaaniASR(config)
        self.chunk_seconds = chunk_seconds

    def transcribe_stream_approx(self, path: str) -> List[str]:
        import soundfile as sf
        import tempfile
        import numpy as np

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        chunk_len = int(self.chunk_seconds * sr)
        if chunk_len <= 0:
            raise ValueError("chunk_seconds must be > 0")

        chunk_paths = []
        tmp_dir = tempfile.mkdtemp(prefix="sravaani_stream_")
        try:
            n_chunks = max(1, int(np.ceil(len(data) / chunk_len)))
            for i in range(n_chunks):
                seg = data[i * chunk_len:(i + 1) * chunk_len]
                if len(seg) == 0:
                    continue
                p = Path(tmp_dir) / f"chunk_{i:04d}.wav"
                sf.write(str(p), seg, sr)
                chunk_paths.append(str(p))
            return self._asr.transcribe_files(chunk_paths)
        finally:
            for p in chunk_paths:
                Path(p).unlink(missing_ok=True)
            Path(tmp_dir).rmdir()


def transcribe_dir_to_jsonl(
    audio_dir: str,
    out_jsonl: str,
    config: Optional[AsrConfig] = None,
    extensions: Iterable[str] = AUDIO_EXTENSIONS,
) -> None:
    """Transcribe every audio file in audio_dir, writing {id,text} JSONL -- the exact schema
    src/eval/compute_wer_delta.py expects for --raw-transcripts / --enhanced-transcripts.
    id = filename stem, so a raw/ dir and an enhanced/ dir with matching filenames (which is
    what src/infer/batch_infer.py produces) line up automatically without any manual mapping.
    """
    audio_dir_p = Path(audio_dir)
    exts = tuple(e.lower() for e in extensions)
    files = sorted(p for p in audio_dir_p.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"No audio files found in {audio_dir} with extensions {exts}")

    asr = SraVaaniASR(config)
    texts = asr.transcribe_files([str(p) for p in files])

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p, text in zip(files, texts):
            f.write(json.dumps({"id": p.stem, "text": text}, ensure_ascii=False) + "\n")


def transcribe_single_file(
    audio_path: str,
    out_path: Optional[str] = None,
    config: Optional[AsrConfig] = None,
) -> dict:
    """Transcribe exactly one audio file. Returns {"id","text"}; if out_path is given, writes
    that same object as a single JSON file (not JSONL -- there's only one record).

    This is the path to use for a single recorded call, a demo, or (via
    src.asr.run_once / the live-enhancement --transcribe flag) the one file that live
    enhancement just produced -- it never scans a directory or transcribes anything else.
    """
    asr = SraVaaniASR(config)
    text = asr.transcribe_file(audio_path)
    result = {"id": Path(audio_path).stem, "text": text}
    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transcribe a directory of call audio with SraVaani.")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--out", required=True, help="Output JSONL path: {id,text} per line")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--hf-token", default=None, help="Overrides HF_TOKEN env var if set")
    args = p.parse_args()

    config = AsrConfig(repo_id=args.repo_id, device=args.device, hf_token=args.hf_token)
    transcribe_dir_to_jsonl(args.audio_dir, args.out, config=config)


if __name__ == "__main__":
    main()
