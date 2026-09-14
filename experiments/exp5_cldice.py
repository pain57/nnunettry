"""Exp 5 — topology-aware loss: Dice+CE vs Dice+CE+clDice.

Compares the official ``nnUNetTrainer`` (Dice+CE) against the custom
``nnUNetTrainer_DiceCEclDice``. The clDice trainer applies a soft-skeletonized
clDice on the full-resolution head and is deep-supervision compatible.
"""

from .common import ensure_custom_trainers_installed, plan_and_preprocess, train


def run(device=None, folds=None):
    plan_and_preprocess()
    ensure_custom_trainers_installed()

    train("nnUNetTrainer", device=device, folds=folds)
    train("nnUNetTrainer_DiceCEclDice", device=device, folds=folds)
