#!/usr/bin/env bash
set -euo pipefail

# Run from voice-enhancement/
python -m src.train.prepare_denoiser_dataset \
  --manifest data/augmented/manifest.jsonl \
  --out-dir data/denoiser_egs/tr

# Clone the archived upstream denoiser repo once:
# git clone https://github.com/facebookresearch/denoiser ../denoiser
# cd ../denoiser
# pip install -r requirements.txt
# Then adapt conf/dset/dns.yaml (or a copy) so its train noisy/clean JSON paths point to:
#   ../voice-enhancement/data/denoiser_egs/tr/noisy.json
#   ../voice-enhancement/data/denoiser_egs/tr/clean.json
# Fine-tune from dns64:
# ./train.py continue_pretrained=dns64 demucs.hidden=64 epochs=40 batch_size=16 lr=3e-5
