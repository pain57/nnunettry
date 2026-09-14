"""Exp 5 — continuity: Dice+CE (baseline) vs Dice+CE+clDice."""

from .common import ensure_custom_trainers_installed, plan_and_preprocess, set_num_epochs, train


def run(device=None, epochs=None):
    plan_and_preprocess()
    if epochs:
        set_num_epochs(epochs)
    ensure_custom_trainers_installed()

    train("nnUNetTrainer", device=device)               # Dice + CE
    train("nnUNetTrainer_DiceCEclDice", device=device)   # Dice + CE + clDice
