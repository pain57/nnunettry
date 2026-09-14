"""Helpers to tweak the official plans file (nnUNetPlans*.json).

These only edit fields that nnU-Net itself reads back (``num_epochs`` and the
architecture init kwargs used by the custom backbones). They never reimplement
planning or preprocessing — both are still done by ``nnUNetv2_plan_and_preprocess``.
"""

from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

from .config import DATASET_NAME, DEFAULT_PLANS, PREPROCESSED_DIR


def _plans_path(dataset_name, plans_identifier):
    return join(PREPROCESSED_DIR, dataset_name, f"{plans_identifier}.json")


def set_num_epochs(epochs, dataset_name=DATASET_NAME, configuration="3d_fullres",
                   plans_identifier=DEFAULT_PLANS):
    """Override the training length recorded in a plans file."""
    plans_path = _plans_path(dataset_name, plans_identifier)
    plans = load_json(plans_path)
    plans["configurations"][configuration]["num_epochs"] = int(epochs)
    save_json(plans, plans_path)
    print(f"set num_epochs={epochs} for {configuration} in {plans_path}")


def patch_arch_init_kwargs(img_size=None, dataset_name=DATASET_NAME,
                           configuration="3d_fullres", plans_identifier=DEFAULT_PLANS):
    """Inject the patch size (``img_size``) into a plans file.

    The custom backbones (UNETR / SwinUNETR) need a fixed input size equal to the
    nnU-Net patch size; their ``build_network_architecture`` reads it from
    ``arch_init_kwargs``. nnU-Net passes the dict stored under one of the keys
    below as ``arch_init_kwargs``, so populate every plausible key to stay
    compatible across minor nnU-Net versions.
    """
    plans_path = _plans_path(dataset_name, plans_identifier)
    plans = load_json(plans_path)
    cfg = plans["configurations"][configuration]
    if img_size is None:
        img_size = cfg["patch_size"]
    img_size = [int(s) for s in img_size]

    for key in ("network_arch_init_kwargs", "arch_init_kwargs"):
        cfg.setdefault(key, {})["img_size"] = img_size
    cfg.setdefault("architecture", {}).setdefault("arch_init_kwargs", {})["img_size"] = img_size
    save_json(plans, plans_path)
    print(f"patched img_size={img_size} into {plans_path}")
