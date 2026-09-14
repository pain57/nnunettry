"""Exp 4 — two-stage pancreas / hepatobiliary ROI -> duct segmentation.

Stage 1 (coarse, 3d_lowres) localizes the pancreas + hepatobiliary ROI on
``Dataset151_PancreasROI``. Stage 2 (fine, 3d_fullres) segments the pancreatic /
bile duct inside the ROI on ``Dataset150_PancreasDuct``. Inference that chains
the two stages lives in ``predict.py --coarse_to_fine``.
"""

from nnunet_utils.config import DATASET_ID, ROI_DATASET_ID

from .common import plan_and_preprocess, train


def run(device=None, folds=None):
    # stage 1: ROI localization (coarse)
    plan_and_preprocess(dataset_id=ROI_DATASET_ID, configurations=("3d_lowres",))
    # stage 2: duct segmentation (fine)
    plan_and_preprocess(dataset_id=DATASET_ID, configurations=("3d_fullres",))

    train("nnUNetTrainer", dataset_id=ROI_DATASET_ID, configuration="3d_lowres",
          device=device, folds=folds)
    train("nnUNetTrainer", dataset_id=DATASET_ID, configuration="3d_fullres",
          device=device, folds=folds)
