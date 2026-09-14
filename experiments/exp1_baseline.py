"""Exp 1 — nnU-Net v2 baseline: default PlainConvUNet, 3d_fullres, Dice+CE."""

from .common import plan_and_preprocess, set_num_epochs, train


def run(device=None, epochs=None):
    plan_and_preprocess()
    if epochs:
        set_num_epochs(epochs)
    train("nnUNetTrainer", device=device)
