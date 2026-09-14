"""Exp 6 — UNETR / SwinUNETR / MedNeXt comparison.

Uses a dedicated plans identifier and patches it to (a) inject the fixed
``img_size`` those backbones need and (b) disable deep supervision, since they
have no DS heads. All three are single-output networks trained with the official
Dice+CE loss (no DS wrapper).
"""

from nnunet_utils.plans_patch import patch_plans

from .common import ensure_custom_trainers_installed, plan_and_preprocess, train

PLANS = "nnUNetPlans_transformer"
TRAINERS = ("nnUNetTrainer_UNETR", "nnUNetTrainer_SwinUNETR", "nnUNetTrainer_MedNeXt")


def run(device=None, folds=None):
    plan_and_preprocess(plans_identifier=PLANS)
    patch_plans(plans_identifier=PLANS)  # img_size + disable deep supervision
    ensure_custom_trainers_installed()

    for trainer in TRAINERS:
        train(trainer, plans_identifier=PLANS, device=device, folds=folds)
