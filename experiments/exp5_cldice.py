"""Experiment 5 — continuity loss.

    Dice + CE  vs  Dice + CE + clDice.

clDice rewards topological continuity of thin / tubular structures by computing
the Dice between soft skeletons. This experiment trains identical models with
and without the clDice term and compares clDice (and Dice) on the validation set.
"""

from nnunet.config import NNUnetConfig
from .common import run_training

LOSSES = ("dice_ce", "dice_ce_cldice")


def build_config(loss: str, **overrides) -> NNUnetConfig:
    cfg = NNUnetConfig()
    cfg.backbone = "plain"
    cfg.num_classes = 2
    cfg.loss = loss
    cfg.num_epochs = 1000
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(data_dir: str = "data", output_dir: str = "runs/exp5_cldice",
        device: str = "cuda", losses=LOSSES, **overrides):
    results = {}
    for loss in losses:
        cfg = build_config(loss, **overrides)
        out = f"{output_dir}/{loss}"
        results[loss] = run_training(cfg, data_dir, out, device=device)
    return results
