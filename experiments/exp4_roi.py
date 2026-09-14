"""Exp 4 — ROI / coarse-to-fine: 3d_lowres localizes, 3d_fullres refines.

nnU-Net already generates a ``3d_lowres`` configuration, which serves as the
coarse stage; the ``3d_fullres`` model is the fine stage. Two-stage inference
(the ROI crop + paste-back) is implemented in ``predict.py --coarse_to_fine``.
"""

from .common import plan_and_preprocess, set_num_epochs, train


def run(device=None, epochs=None):
    plan_and_preprocess(configurations=("3d_fullres", "3d_lowres"))
    if epochs:
        set_num_epochs(epochs, configuration="3d_fullres")
        set_num_epochs(epochs, configuration="3d_lowres")

    train("nnUNetTrainer", configuration="3d_lowres", device=device)   # coarse (localization)
    train("nnUNetTrainer", configuration="3d_fullres", device=device)  # fine (segmentation)

    print("\nTwo-stage inference:  python predict.py --coarse_to_fine --input <images> --output <out>")
