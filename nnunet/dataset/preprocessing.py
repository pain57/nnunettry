"""Intensity normalization, resampling and cropping utilities."""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import zoom


def ct_intensity_normalization(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """CT normalization: clip to percentile range then z-score the foreground."""
    p_low = np.percentile(image, lower_percentile)
    p_high = np.percentile(image, upper_percentile)
    image = np.clip(image, p_low, p_high)
    mask = image > p_low
    if mask.any():
        mean = image[mask].mean()
        std = image[mask].std()
        std = std if std > 0 else 1.0
        image = (image - mean) / std
    else:
        image = image - image.mean()
    return image


def zscore_normalization(image: np.ndarray) -> np.ndarray:
    """Standard z-score normalization across the whole volume."""
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    return image


def mri_intensity_normalization(image: np.ndarray) -> np.ndarray:
    """MRI normalization (per-volume z-score of non-zero foreground)."""
    p_low = np.percentile(image, 0.5)
    p_high = np.percentile(image, 99.5)
    image = np.clip(image, p_low, p_high)
    mask = image > 0
    if mask.any():
        mean = image[mask].mean()
        std = image[mask].std()
    else:
        mean = image.mean()
        std = image.std()
    std = std if std > 0 else 1.0
    return (image - mean) / std


def normalize_volume(image: np.ndarray, modality: str = "ct") -> np.ndarray:
    """Dispatch intensity normalization by modality ('ct' or 'mri')."""
    if modality == "mri":
        return mri_intensity_normalization(image)
    return ct_intensity_normalization(image)


def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    is_label: bool = False,
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D volume to a target voxel spacing.

    Labels use nearest-neighbor interpolation (order=0); images use the given
    order (1=linear, 3=cubic).
    """
    zoom_factors = tuple(o / t for o, t in zip(original_spacing, target_spacing))
    if is_label:
        return zoom(volume, zoom_factors, order=0, prefilter=False)
    return zoom(volume, zoom_factors, order=order)


def resample_to_target_spacing(
    image: np.ndarray,
    mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order_image: int = 3,
    order_label: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an image/label pair to a common target spacing (Experiment 3)."""
    image = resample_volume(image, original_spacing, target_spacing, is_label=False, order=order_image)
    mask = resample_volume(mask, original_spacing, target_spacing, is_label=True, order=order_label)
    return image, mask


def foreground_crop(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    margin: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[slice, slice, slice]]:
    """Crop to the foreground bounding box with a margin."""
    if mask is not None:
        fg = mask > 0
    else:
        fg = image > 0

    if not fg.any():
        return image, mask, (slice(None), slice(None), slice(None))

    coords = np.argwhere(fg)
    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_min, x_max = coords[:, 2].min(), coords[:, 2].max()

    z_min = max(0, z_min - margin)
    z_max = min(image.shape[0], z_max + margin + 1)
    y_min = max(0, y_min - margin)
    y_max = min(image.shape[1], y_max + margin + 1)
    x_min = max(0, x_min - margin)
    x_max = min(image.shape[2], x_max + margin + 1)

    crop = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    image_cropped = image[crop].copy()
    mask_cropped = mask[crop].copy() if mask is not None else None
    return image_cropped, mask_cropped, crop


def pad_to_patch_size(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Pad a volume so each spatial dimension is at least patch_size."""
    original_shape = volume.shape
    pad_dims = []
    for s, p in zip(volume.shape, patch_size):
        if s < p:
            diff = p - s
            pad_dims.append((diff // 2, diff - diff // 2))
        else:
            pad_dims.append((0, 0))
    padded = np.pad(volume, pad_dims, mode="constant", constant_values=0)
    return padded, original_shape
