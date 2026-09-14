"""Exp 6 — compare UNETR / SwinUNETR / MedNeXt against the baseline."""

from nnunet_utils.plans_patch import patch_arch_init_kwargs

from .common import ensure_custom_trainers_installed, plan_and_preprocess, set_num_epochs, train


def run(device=None, epochs=None):
    # Dedicated plans file so injecting ``img_size`` never disturbs the baseline
    # ``nnUNetPlans.json`` used by Experiments 1-5.
    plan_and_preprocess(plans_identifier="nnUNetPlans_transformer")
    if epochs:
        set_num_epochs(epochs, plans_identifier="nnUNetPlans_transformer")

    ensure_custom_trainers_installed()
    patch_arch_init_kwargs(plans_identifier="nnUNetPlans_transformer")

    for name in ("nnUNetTrainer_UNETR", "nnUNetTrainer_SwinUNETR", "nnUNetTrainer_MedNeXt"):
        train(name, plans_identifier="nnUNetPlans_transformer", device=device)
