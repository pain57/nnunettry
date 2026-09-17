"""Exp 3 — target spacing study (AMOS MRI: observe the auto spacing first).

By default **no fixed spacing** is imposed: nnU-Net decides the target spacing
from the actual AMOS MRI voxel sizes, and this script reads it back from the
generated plans file and reports it. A fixed finer spacing can be added later
(``target_spacing``) once the automatic value is known — do not hard-code
``(1, 1, 1)`` mm.
"""

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunet_utils.config import DATASET_NAME, DEFAULT_PLANS, PREPROCESSED_DIR

from .common import plan_and_preprocess, train

FINE_PLANS = "nnUNetPlans_fine"


def _report_auto_spacing(plans_identifier=DEFAULT_PLANS):
    plans = load_json(join(PREPROCESSED_DIR, DATASET_NAME, f"{plans_identifier}.json"))
    cfg = plans["configurations"]["3d_fullres"]
    print(f"\n[Exp3] nnU-Net auto target spacing = {cfg['spacing']} mm")
    print(f"[Exp3] median image size after resampling = {cfg.get('median_image_size_in_voxels')}")
    return tuple(cfg["spacing"])


def run(device=None, folds=None, target_spacing=None):
    plan_and_preprocess()
    _report_auto_spacing()

    if target_spacing is not None:
        plan_and_preprocess(plans_identifier=FINE_PLANS, target_spacing=target_spacing)
        train("nnUNetTrainer", plans_identifier=DEFAULT_PLANS, device=device, folds=folds)
        train("nnUNetTrainer", plans_identifier=FINE_PLANS, device=device, folds=folds)
    else:
        train("nnUNetTrainer", plans_identifier=DEFAULT_PLANS, device=device, folds=folds)
