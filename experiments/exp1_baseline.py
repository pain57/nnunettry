"""Exp 1 — nnU-Net v2 3D fullres baseline for AMOS22 MRI pancreas segmentation.

Official planner + official ``nnUNetTrainer`` (PlainConvUNet, Dice+CE, 1000
epochs), no changes. Trains all 5 folds.
"""

from .common import plan_and_preprocess, train


def run(device=None, folds=None):
    plan_and_preprocess()
    train("nnUNetTrainer", device=device, folds=folds)
