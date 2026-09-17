"""Exp 4 — two-stage pancreas/ROI -> duct segmentation.  *** PAUSED ***

Not runnable until pancreatic-duct / bile-duct ground truth is available (the
current dataset is single-organ AMOS MRI pancreas). Kept for the future duct
work; ``run_experiment.py`` does not map ``--exp 4``.
"""


def run(device=None, folds=None):
    raise NotImplementedError(
        "Exp 4 (ROI -> duct, two-stage) is paused: no duct/ROI ground truth yet. "
        "Re-enable once the pancreatic-duct / bile-duct dataset is available."
    )
