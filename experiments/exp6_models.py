"""Exp 6 — UNETR / SwinUNETR / MedNeXt comparison.

Deep supervision is disabled for these single-output backbones (they have no DS
heads): the plans flag is turned off via ``disable_deep_supervision``, and each
trainer's ``_build_loss`` returns the plain Dice+CE without the DS wrapper. Input
size is read directly from ``configuration_manager.patch_size`` inside the
network-building method (2.8.1 trainer interface).
"""

from nnunet_utils.plans_patch import disable_deep_supervision

from .common import ensure_custom_trainers_installed, plan_and_preprocess, train

PLANS = "nnUNetPlans_transformer"
TRAINERS = ("nnUNetTrainer_UNETR", "nnUNetTrainer_SwinUNETR", "nnUNetTrainer_MedNeXt")


def run(device=None, folds=None):
    plan_and_preprocess(plans_identifier=PLANS)
    disable_deep_supervision(plans_identifier=PLANS)
    ensure_custom_trainers_installed()

    for trainer in TRAINERS:
        train(trainer, plans_identifier=PLANS, device=device, folds=folds)
