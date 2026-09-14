"""Exp 3 — target spacing: default vs a finer (isotropic) target spacing.

Uses the official ``--overwrite_target_spacing`` flag to produce a second set of
plans that resamples to the requested spacing, keeping the baseline plans intact.
"""

from .common import plan_and_preprocess, set_num_epochs, train


def run(device=None, epochs=None, spacing=(1.0, 1.0, 1.0)):
    # baseline plans at the automatically-determined spacing
    plan_and_preprocess(plans_identifier="nnUNetPlans")
    # separate plans that resample to the finer target spacing
    plan_and_preprocess(plans_identifier="nnUNetPlans_1mm", target_spacing=spacing)

    if epochs:
        set_num_epochs(epochs, plans_identifier="nnUNetPlans")
        set_num_epochs(epochs, plans_identifier="nnUNetPlans_1mm")

    train("nnUNetTrainer", plans_identifier="nnUNetPlans", device=device)
    train("nnUNetTrainer", plans_identifier="nnUNetPlans_1mm", device=device)
