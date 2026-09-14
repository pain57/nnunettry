"""Exp 4 — ROI localization (coarse stage validation).

For now this is *not* the full two-stage coarse-to-fine: it validates that a
coarse (3d_lowres) model can localize the target region. Once you have real
pancreatic-duct / bile-duct GT, turn this into the true two-stage pipeline
(coarse localize pancreas ROI -> fine segment ducts inside the ROI).

Evaluate the localization with::

    python evaluate.py --pred_dir <lowres_preds> --gt_dir <labelsTs> --localization
"""

from .common import plan_and_preprocess, train


def run(device=None, folds=None):
    plan_and_preprocess(configurations=("3d_fullres", "3d_lowres"))
    # coarse stage: locate the ROI (3d_lowres, larger field of view)
    train("nnUNetTrainer", configuration="3d_lowres", device=device, folds=folds)
    # fine stage: kept for the future two-stage (full-res within the ROI)
    train("nnUNetTrainer", configuration="3d_fullres", device=device, folds=folds)
