import json
from pathlib import Path


def test_denoiser_manifest_shape(tmp_path):
    import soundfile as sf
    import numpy as np
    from src.train.prepare_denoiser_dataset import main

    clean = tmp_path / "clean.wav"
    noisy = tmp_path / "noisy.wav"
    sf.write(clean, np.zeros(160, dtype=np.float32), 16000)
    sf.write(noisy, np.zeros(160, dtype=np.float32), 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"clean": str(clean), "noisy": str(noisy)}) + "\n")
    out = tmp_path / "egs"

    import sys
    old = sys.argv
    try:
        sys.argv = ["prepare_denoiser_dataset", "--manifest", str(manifest), "--out-dir", str(out)]
        main()
    finally:
        sys.argv = old

    clean_json = json.loads((out / "clean.json").read_text())
    noisy_json = json.loads((out / "noisy.json").read_text())
    assert clean_json == [[str(clean), 160]]
    assert noisy_json == [[str(noisy), 160]]


def test_pooled_wer():
    from src.eval.compute_wer_delta import pooled_wer
    assert pooled_wer(["a b", "a"], ["a b", "x"]) == 1 / 3
