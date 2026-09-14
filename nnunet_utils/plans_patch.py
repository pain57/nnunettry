"""Helpers to tweak the official plans file (nnUNetPlans*.json).

These only edit fields that nnU-Net itself reads back (architecture init kwargs
and the deep-supervision flag). Planning and preprocessing are still done by
``nnUNetv2_plan_and_preprocess``. Training length is left at the official
default (1000 epochs) — it is no longer overridden via the plans file.
"""

from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

from .config import DATASET_NAME, DEFAULT_PLANS, PREPROCESSED_DIR


def _plans_path(dataset_name, plans_identifier):
    return join(PREPROCESSED_DIR, dataset_name, f"{plans_identifier}.json")


def patch_plans(img_size=None, enable_deep_supervision=False, dataset_name=DATASET_NAME,
                configuration="3d_fullres", plans_identifier=DEFAULT_PLANS):
    """Prepare a plans file for the single-output backbones (Exp 6).

    * injects ``img_size`` (== patch size) into the architecture init kwargs so
      UNETR / SwinUNETR get the correct fixed input size;
    * disables deep supervision, since those backbones have no DS heads (this
      keeps the data loader's single target, the loss and the network consistent).
    """
    plans_path = _plans_path(dataset_name, plans_identifier)
    plans = load_json(plans_path)
    cfg = plans["configurations"][configuration]
    if img_size is None:
        img_size = cfg["patch_size"]
    img_size = [int(s) for s in img_size]

    # nnU-Net passes one of these dicts to build_network_architecture(...) as
    # ``arch_init_kwargs``; populate every plausible key to stay compatible
    # across minor nnU-Net versions.
    for key in ("network_arch_init_kwargs", "arch_init_kwargs"):
        cfg.setdefault(key, {})["img_size"] = img_size
    cfg.setdefault("architecture", {}).setdefault("arch_init_kwargs", {})["img_size"] = img_size

    cfg["enable_deep_supervision"] = bool(enable_deep_supervision)
    save_json(plans, plans_path)
    print(f"patched {plans_path}: img_size={img_size}, "
          f"enable_deep_supervision={enable_deep_supervision}")
