import json
from pathlib import Path

import pytest

from src.asr.transcribe import AsrConfig, SraVaaniASR, transcribe_dir_to_jsonl


def test_asr_config_missing_token_raises_clear_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = AsrConfig()
    with pytest.raises(RuntimeError, match="Hugging Face token"):
        config.resolve_token()


def test_asr_config_explicit_token_bypasses_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = AsrConfig(hf_token="explicit-token")
    assert config.resolve_token() == "explicit-token"


def test_transcribe_dir_to_jsonl_writes_expected_schema(tmp_path, monkeypatch):
    """Verifies the {id,text} JSONL contract compute_wer_delta.py depends on, using a stub
    ASR so this test needs neither network access nor real model weights."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "call_001.wav").write_bytes(b"fake wav bytes")
    (audio_dir / "call_002.wav").write_bytes(b"fake wav bytes")
    (audio_dir / "not_audio.txt").write_text("ignored")

    def fake_transcribe_files(self, paths):
        return [f"stub transcript for {Path(p).stem}" for p in paths]

    monkeypatch.setattr(SraVaaniASR, "transcribe_files", fake_transcribe_files)

    out_jsonl = tmp_path / "out.jsonl"
    transcribe_dir_to_jsonl(str(audio_dir), str(out_jsonl))

    lines = [json.loads(l) for l in out_jsonl.read_text().splitlines()]
    ids = {r["id"] for r in lines}
    assert ids == {"call_001", "call_002"}
    assert all("text" in r for r in lines)


def test_transcribe_dir_to_jsonl_no_audio_files_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit):
        transcribe_dir_to_jsonl(str(empty_dir), str(tmp_path / "out.jsonl"))


def test_transcribe_dir_to_jsonl_strips_prefix(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "new_call_001.wav").write_bytes(b"fake wav bytes")
    (audio_dir / "new_call_002.wav").write_bytes(b"fake wav bytes")

    def fake_transcribe_files(self, paths):
        return [f"stub transcript for {Path(p).name}" for p in paths]

    monkeypatch.setattr(SraVaaniASR, "transcribe_files", fake_transcribe_files)

    out_jsonl = tmp_path / "out.jsonl"
    transcribe_dir_to_jsonl(str(audio_dir), str(out_jsonl), prefix_to_strip="new_")

    lines = [json.loads(l) for l in out_jsonl.read_text().splitlines()]
    ids = {r["id"] for r in lines}
    assert ids == {"call_001", "call_002"}

