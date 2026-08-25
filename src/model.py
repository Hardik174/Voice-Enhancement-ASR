from __future__ import annotations

import torch


def load_model(model_name: str = "dns64", checkpoint: str | None = None, device: str = "cpu"):
    try:
        from denoiser import pretrained
    except ImportError as exc:
        raise RuntimeError("Install the upstream speech denoiser first: pip install denoiser") from exc

    if checkpoint:
        pkg = torch.load(checkpoint, map_location="cpu")
        model = pretrained.get_model(type("Args", (), {"model_path": checkpoint, "dns48": False, "dns64": False, "master64": False, "valentini_nc": False})())
    else:
        factories = {"dns48": pretrained.dns48, "dns64": pretrained.dns64, "master64": pretrained.master64}
        if model_name not in factories:
            raise ValueError(f"Unsupported model: {model_name}")
        model = factories[model_name](pretrained=True)

    model = model.to(device).eval()
    return model
