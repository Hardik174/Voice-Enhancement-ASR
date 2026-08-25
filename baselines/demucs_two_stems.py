from __future__ import annotations
import argparse
import demucs.separate

def main():
    p=argparse.ArgumentParser(description="Throwaway music-Demucs vocals baseline; not production speech denoising.")
    p.add_argument("input"); p.add_argument("--out-dir",default="separated")
    a=p.parse_args()
    demucs.separate.main(["--two-stems","vocals","-n","htdemucs","-o",a.out_dir,a.input])

if __name__ == "__main__": main()
