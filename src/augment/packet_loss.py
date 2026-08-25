from __future__ import annotations
import random
import torch


def gilbert_elliott_loss(wav: torch.Tensor, loss_rate: float, frame_ms: int = 20, sr: int = 16000) -> torch.Tensor:
    """Simple bursty packet-loss simulation with fade/hold PLC."""
    if loss_rate <= 0:
        return wav.clone()
    x = wav.flatten().clone()
    frame = max(1, int(sr * frame_ms / 1000))
    bad = False
    out = x.clone()
    i = 0
    while i < len(x):
        if bad:
            bad = random.random() > 0.35
        else:
            bad = random.random() < loss_rate
        j = min(len(x), i + frame)
        if bad:
            prev = out[i - 1] if i else torch.tensor(0.0, device=x.device)
            out[i:j] = torch.linspace(prev, 0, j - i, device=x.device)
        else:
            out[i:j] = x[i:j]
        i = j
    return out.reshape_as(wav)
