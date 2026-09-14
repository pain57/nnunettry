"""Experiment 1 — establish the baseline.

    nnU-Net v2 · 3D fullres · default PlainConvUNet.

Runs the plain convolutional U-Net with the default Dice+CE loss on the native
spacing, producing the reference checkpoint every later experiment is compared
against.
"""

from nnunet.config import NNUnetConfig
from .common import run_training


def build_config(**overrides) -> NNUnetConfig:
    cfg = NNUnetConfig()
    cfg.backbone = "plain"
    cfg.num_classes = 2
    cfg.loss = "dice_ce"
    cfg.num_epochs = 1000
    cfg.target_spacing = None          # native spacing
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(data_dir: str = "data", output_dir: str = "runs/exp1_baseline",
        device: str = "cuda", **overrides):
    cfg = build_config(**overrides)
    return run_training(cfg, data_dir, output_dir, device=device)
