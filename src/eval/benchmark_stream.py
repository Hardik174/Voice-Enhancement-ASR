from __future__ import annotations
import argparse, time
import torch
from src.model import load_model
from denoiser.demucs import DemucsStreamer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='dns64'); p.add_argument('--seconds',type=float,default=10); p.add_argument('--threads',type=int,default=4); a=p.parse_args()
    torch.set_num_threads(a.threads); model=load_model(a.model, device='cpu'); s=DemucsStreamer(model,dry=.01)
    x=torch.randn(1,int(16000*a.seconds)); i=0; total=0; start=time.perf_counter(); first=s.total_length
    with torch.inference_mode():
        while i < x.shape[1]:
            n=first if i==0 else s.stride
            t=time.perf_counter(); s.feed(x[:,i:i+n]); total+=time.perf_counter()-t; i+=n
        s.flush()
    audio=max(1e-9,a.seconds); print(f'RTF={total/audio:.3f} compute_seconds={total:.3f} audio_seconds={audio:.3f} stride_ms={s.stride/16:.1f}')

if __name__=='__main__': main()
