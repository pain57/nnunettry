"""Exp 2 — compare ResEnc M / L backbones (official nnU-Net v2 variants)."""

from .common import plan_and_preprocess, set_num_epochs, train


def run(device=None, epochs=None):
    plan_and_preprocess()
    if epochs:
        set_num_epochs(epochs)
    for name in ("nnUNetTrainerResEncM", "nnUNetTrainerResEncL"):
        train(name, device=device)
