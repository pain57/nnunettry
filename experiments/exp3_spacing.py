"""Exp 3 — target spacing study for thin duct structures.

Planning with ``--overwrite_target_spacing`` produces a second set of plans +
preprocessed data (finer spacing) alongside the default baseline plans. Run this
with the spacing measured from your real data, e.g. ``spacing=(0.75, 0.75, 0.75)``.
"""

from .common import DEFAULT_PLANS, plan_and_preprocess, train

FINE_PLANS = "nnUNetPlans_fine"


def run(device=None, folds=None, spacing=(1.0, 1.0, 1.0)):
    # baseline (default planner, data-driven spacing)
    plan_and_preprocess()
    # finer fixed spacing for the sub-mm ducts
    plan_and_preprocess(plans_identifier=FINE_PLANS, target_spacing=spacing)

    train("nnUNetTrainer", plans_identifier=DEFAULT_PLANS, device=device, folds=folds)
    train("nnUNetTrainer", plans_identifier=FINE_PLANS, device=device, folds=folds)
