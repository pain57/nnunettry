"""Dataset loading / preprocessing shared by train.py and the experiment scripts.

Loads a ``dataset.json`` following the nnU-Net convention, applies intensity
normalization, optional target-spacing resampling (Experiment 3) and
foreground cropping.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from nnunet.config import NNUnetConfig
from nnunet.utils.nifti_io import load_nifti, load_nifti_with_spacing
from nnunet.dataset.preprocessing import (
    normalize_volume,
    resample_volume,
    foreground_crop,
)


def _parse_case_id(image_path: str) -> int:
    """Best-effort case-id extraction (used for CT/MRI modality filtering)."""
    try:
        name = image_path.split("/")[-1].replace(".nii.gz", "")
        return int(name.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def preprocess_case(
    image_path: str,
    mask_path: str,
    config: NNUnetConfig,
    pancreas_only: bool = False,
    modality: str = "ct",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess a single image/label pair.

    Returns (image, mask, affine) with the same voxel spacing as the target
    spacing in ``config`` (or native spacing if ``config.target_spacing`` is None).
    """
    image, affine, spacing = load_nifti_with_spacing(image_path)
    mask, _, _ = load_nifti_with_spacing(mask_path)

    image = normalize_volume(image, modality=modality)

    if pancreas_only:
        mask = (mask == 10).astype(np.int64)

    # Experiment 3: resample to the requested target spacing.
    if config.target_spacing is not None:
        image = resample_volume(image, spacing, config.target_spacing,
                                is_label=False, order=config.resample_order_image)
        mask = resample_volume(mask, spacing, config.target_spacing,
                               is_label=True, order=config.resample_order_label)

    image, mask, _ = foreground_crop(image, mask, margin=30)

    return image.astype(np.float32), mask.astype(np.int64), affine


def load_dataset(
    data_dir: str,
    config: NNUnetConfig,
    mode: str = "train",
    pancreas_only: bool = False,
    modality: str = "ct",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load all cases of a split.

    Returns (images, masks, affines). Masks are empty for ``mode="test"``.
    """
    dataset_json_path = Path(data_dir) / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(
            f"dataset.json not found in {data_dir}. Run generate_synthetic_data.py first."
        )

    with open(dataset_json_path) as f:
        dataset_info = json.load(f)

    base_dir = Path(data_dir)
    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    affines: List[np.ndarray] = []

    if mode in ("train", "validation"):
        key = "training" if mode == "train" else "validation"
        cases = dataset_info.get(key, [])

        for case in cases:
            img_path = base_dir / case["image"]
            lbl_path = base_dir / case["label"]

            case_id = _parse_case_id(case["image"])
            is_mri = case_id >= 500

            if modality == "ct" and is_mri:
                print(f"  Skipped (MRI): {img_path.name}")
                continue
            if modality == "mri" and not is_mri:
                print(f"  Skipped (CT): {img_path.name}")
                continue

            if img_path.exists() and lbl_path.exists():
                img, msk, affine = preprocess_case(
                    str(img_path), str(lbl_path), config,
                    pancreas_only=pancreas_only, modality=modality,
                )
                images.append(img)
                masks.append(msk)
                affines.append(affine)
                tag = "MRI" if is_mri else "CT"
                print(f"  Loaded [{tag}]: {img_path.name} | shape: {img.shape} | mask voxels: {int(msk.sum())}")

    elif mode == "test":
        cases = dataset_info.get("test", [])
        for case in cases:
            img_path = base_dir / (case["image"] if isinstance(case, dict) else case)
            if img_path.exists():
                img, affine, _ = load_nifti_with_spacing(str(img_path))
                img = normalize_volume(img, modality=modality)
                images.append(img.astype(np.float32))
                affines.append(affine)
                print(f"  Loaded test: {img_path.name} | shape: {img.shape}")

    return images, masks, affines
