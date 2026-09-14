"""Experiment 3 — target spacing.

    Adjust target spacing -> study micro-vessel / thin-duct resolution.

The native CT spacing is anisotropic (e.g. 1.5 x 1.0 x 1.0 mm). This experiment
trains the same baseline model at several resampled spacings and compares Dice /
clDice, since finer isotropic spacing can preserve thin tubular structures at
the cost of more memory and compute.
"""

from nnunet.config import NNUnetConfig
from .common import run_training

# name -> target voxel spacing (mm). None = keep native spacing.
SPACINGS = {
    "native": None,
    "iso_1.00mm": (1.0, 1.0, 1.0),
    "iso_0.75mm": (0.75, 0.75, 0.75),
    "aniso_1.0_1.0_2.0": (1.0, 1.0, 2.0),
}


def build_config(target_spacing, **overrides) -> NNUnetConfig:
    cfg = NNUnetConfig()
    cfg.backbone = "plain"
    cfg.num_classes = 2
    cfg.loss = "dice_ce"
    cfg.num_epochs = 1000
    cfg.target_spacing = target_spacing
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(data_dir: str = "data", output_dir: str = "runs/exp3_spacing",
        device: str = "cuda", spacings=SPACINGS, **overrides):
    results = {}
    for name, spacing in spacings.items():
        cfg = build_config(spacing, **overrides)
        out = f"{output_dir}/{name}"
        results[name] = run_training(cfg, data_dir, out, device=device)
    return results
