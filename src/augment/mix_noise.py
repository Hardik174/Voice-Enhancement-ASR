from __future__ import annotations
import math
import random
import torch


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x).clamp_min(1e-12))


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    n = noise.flatten()
    c = clean.flatten()
    if n.numel() < c.numel():
        reps = math.ceil(c.numel() / n.numel())
        n = n.repeat(reps)
    start = random.randint(0, max(0, n.numel() - c.numel()))
    n = n[start:start + c.numel()]
    scale = rms(c) / (rms(n) * (10 ** (snr_db / 20)))
    return (c + scale * n).clamp(-1, 1).reshape_as(clean)
