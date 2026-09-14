"""Helpers to tweak the official plans file (nnUNetPlans*.json).

The only plans edit left is disabling deep supervision for the single-output
backbones (Exp 6). Training length is left at the official default (1000 epochs)
and is no longer overridden via the plans file.
"""

from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

from .config import DATASET_NAME, DEFAULT_PLANS, PREPROCESSED_DIR


def _plans_path(dataset_name, plans_identifier):
    return join(PREPROCESSED_DIR, dataset_name, f"{plans_identifier}.json")


def disable_deep_supervision(dataset_name=DATASET_NAME, configuration="3d_fullres",
                             plans_identifier=DEFAULT_PLANS):
    """Disable deep supervision in a plans file (Exp 6).

    The single-output backbones (UNETR / SwinUNETR / MedNeXt) have no DS heads;
    turning the flag off keeps the data loader's single target, the network and
    the loss consistent.
    """
    plans_path = _plans_path(dataset_name, plans_identifier)
    plans = load_json(plans_path)
    plans["configurations"][configuration]["enable_deep_supervision"] = False
    save_json(plans, plans_path)
    print(f"disabled deep supervision in {plans_path}")
