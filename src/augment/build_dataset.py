from __future__ import annotations
import argparse, json, random
from pathlib import Path
import yaml
import torch
import torchaudio
from src.augment.mix_noise import mix_at_snr
from src.augment.packet_loss import gilbert_elliott_loss
from src.augment.codec_sim import ffmpeg_codec

EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

def files(dirs):
    out=[]
    for d in dirs:
        p=Path(d)
        if p.exists(): out += [x for x in p.rglob("*") if x.suffix.lower() in EXTS]
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data_augmentation.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--max-examples", type=int, default=0)
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    sr=cfg.get("sample_rate",16000)
    clean_files=files(cfg["clean_dirs"]); noise_files=files(cfg["noise_dirs"])
    if not clean_files: raise SystemExit("No clean audio found in configured clean_dirs")
    if not noise_files: raise SystemExit("No noise audio found in configured noise_dirs")
    out=Path(args.out or cfg["output_manifest"]); out.parent.mkdir(parents=True,exist_ok=True)
    n=args.max_examples or len(clean_files)
    snr_lo,snr_hi=cfg["snr_db_range"]
    rates=cfg["packet_loss"]["rates"]
    codecs=cfg.get("codecs",["none"])
    with out.open("w",encoding="utf-8") as f:
        for i in range(n):
            cp=clean_files[i % len(clean_files)]; npth=random.choice(noise_files)
            clean, clean_sr=torchaudio.load(str(cp)); clean=clean.mean(0,keepdim=True)
            clean=torchaudio.functional.resample(clean, clean_sr, sr) if clean_sr != sr else clean
            noise,nsr=torchaudio.load(str(npth)); noise=noise.mean(0,keepdim=True)
            noise=torchaudio.functional.resample(noise,nsr,sr) if nsr!=sr else noise
            snr=random.uniform(snr_lo,snr_hi); loss=random.choice(rates); codec=random.choice(codecs)
            noisy=mix_at_snr(clean,noise,snr)
            noisy=gilbert_elliott_loss(noisy,loss, cfg["packet_loss"].get("frame_ms",20),sr)
            noisy=ffmpeg_codec(noisy,codec,sr)
            base=out.parent / f"sample_{i:07d}"
            clean_path=base.with_name(base.name+"_clean.wav"); noisy_path=base.with_name(base.name+"_noisy.wav")
            torchaudio.save(str(clean_path),clean,sr); torchaudio.save(str(noisy_path),noisy,sr)
            f.write(json.dumps({"clean":str(clean_path),"noisy":str(noisy_path),"snr_db":snr,"packet_loss_rate":loss,"codec":codec})+"\n")
    print(f"Wrote {n} examples to {out}")

if __name__ == "__main__": main()
