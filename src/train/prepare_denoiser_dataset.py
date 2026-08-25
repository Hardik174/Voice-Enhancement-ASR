from __future__ import annotations
import argparse, json
import soundfile as sf
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="Convert our JSONL manifest into denoiser clean.json/noisy.json files.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", default="data/denoiser_egs/tr")
    args=p.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    clean=[]; noisy=[]
    with open(args.manifest,encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); clean_len=sf.info(r["clean"]).frames; noisy_len=sf.info(r["noisy"]).frames; clean.append([r["clean"], clean_len]); noisy.append([r["noisy"], noisy_len])
    (out/"clean.json").write_text(json.dumps(clean,indent=2))
    (out/"noisy.json").write_text(json.dumps(noisy,indent=2))
    print(f"Wrote {len(clean)} clean/noisy entries to {out}")

if __name__ == "__main__": main()
