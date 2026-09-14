"""Experiment 2 — compare backbones.

    nnU-Net v2 · ResEnc M / L · 3D fullres.

Trains the residual-encoder U-Net in its M and L sizes and compares them against
the plain baseline. The only difference between M and L is the encoder channel
depth / block count (see ``nnunet.network.resenc``).
"""

from nnunet.config import NNUnetConfig
from .common import run_training

SIZES = ("m", "l")


def build_config(size: str, **overrides) -> NNUnetConfig:
    cfg = NNUnetConfig()
    cfg.backbone = f"resenc_{size}"
    cfg.num_classes = 2
    cfg.loss = "dice_ce"
    cfg.num_epochs = 1000
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(data_dir: str = "data", output_dir: str = "runs/exp2_backbone",
        device: str = "cuda", sizes=SIZES, **overrides):
    results = {}
    for size in sizes:
        cfg = build_config(size, **overrides)
        out = f"{output_dir}/resenc_{size}"
        results[size] = run_training(cfg, data_dir, out, device=device)
    return results
