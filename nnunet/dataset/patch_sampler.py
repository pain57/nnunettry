"""Patch-based 3D data sampling with optional ROI focusing (Experiment 4)."""

import torch
import numpy as np
from typing import Tuple, Optional, List


def _crop_or_pad(volume: np.ndarray, d0: int, h0: int, w0: int,
                 patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Crop a patch at (d0,h0,w0) and zero-pad it to exactly patch_size."""
    pD, pH, pW = patch_size
    D, H, W = volume.shape
    patch = volume[d0:d0 + pD, h0:h0 + pH, w0:w0 + pW]
    if patch.shape != patch_size:
        padded = np.zeros(patch_size, dtype=patch.dtype)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded
    return patch


def get_roi_bounds(mask: np.ndarray, margin: int = 16) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Bounding box of the foreground (pancreas) expanded by margin.

    Returns ((z_min, z_max), (y_min, y_max), (x_min, x_max)) or None if empty.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    D, H, W = mask.shape
    return (
        (max(0, z_min - margin), min(D, z_max + margin + 1)),
        (max(0, y_min - margin), min(H, y_max + margin + 1)),
        (max(0, x_min - margin), min(W, x_max + margin + 1)),
    )


def get_random_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    force_fg: bool = False,
    fg_min_ratio: float = 0.01,
    roi_bounds: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random patch (optionally within an ROI / forced foreground)."""
    pD, pH, pW = patch_size

    def sample_once(bounds):
        (z0, z1), (y0, y1), (x0, x1) = bounds
        d0 = np.random.randint(z0, max(z0 + 1, z1 - pD)) if z1 - pD > z0 else z0
        h0 = np.random.randint(y0, max(y0 + 1, y1 - pH)) if y1 - pH > y0 else y0
        w0 = np.random.randint(x0, max(x0 + 1, x1 - pW)) if x1 - pW > x0 else x0
        return _crop_or_pad(volume, d0, h0, w0, patch_size), \
               _crop_or_pad(mask, d0, h0, w0, patch_size), (d0, h0, w0)

    D, H, W = volume.shape
    full_bounds = ((0, D), (0, H), (0, W))
    bounds = roi_bounds if roi_bounds is not None else full_bounds

    if force_fg and mask.sum() > 0:
        for _ in range(50):
            img_patch, msk_patch, _ = sample_once(bounds)
            fg_ratio = msk_patch.sum() / msk_patch.size
            if fg_ratio >= fg_min_ratio:
                return img_patch, msk_patch, (0, 0, 0)

    img_patch, msk_patch, start = sample_once(bounds)
    return img_patch, msk_patch, start


class PatchSampler:
    """Yields (image_patch, mask_patch) pairs with foreground balancing."""

    def __init__(
        self,
        volumes: List[np.ndarray],
        masks: List[np.ndarray],
        patch_size: Tuple[int, int, int],
        samples_per_volume: int = 8,
        force_fg_ratio: float = 0.33,
        fg_min_ratio: float = 0.01,
        use_roi: bool = False,
        roi_margin: int = 16,
    ):
        self.volumes = volumes
        self.masks = masks
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.force_fg_ratio = force_fg_ratio
        self.fg_min_ratio = fg_min_ratio
        self.use_roi = use_roi
        self.roi_margin = roi_margin
        self.num_samples = samples_per_volume * len(volumes)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        return self._generate()

    def _generate(self):
        for vol_idx in range(len(self.volumes)):
            volume = self.volumes[vol_idx]
            mask = self.masks[vol_idx]
            roi_bounds = get_roi_bounds(mask, self.roi_margin) if self.use_roi else None
            num_fg_patches = int(self.samples_per_volume * self.force_fg_ratio)

            for i in range(self.samples_per_volume):
                force_fg = i < num_fg_patches
                img_patch, msk_patch, _ = get_random_patch(
                    volume, mask, self.patch_size,
                    force_fg=force_fg, fg_min_ratio=self.fg_min_ratio,
                    roi_bounds=roi_bounds,
                )
                yield img_patch, msk_patch


class PatchDataset3D(torch.utils.data.IterableDataset):
    """Torch IterableDataset for patch-based 3D training (with ROI option)."""

    def __init__(
        self,
        volumes: List[np.ndarray],
        masks: List[np.ndarray],
        patch_size: Tuple[int, int, int],
        samples_per_volume: int = 8,
        force_fg_ratio: float = 0.33,
        fg_min_ratio: float = 0.01,
        use_roi: bool = False,
        roi_margin: int = 16,
        augment_fn=None,
    ):
        super().__init__()
        self.volumes = volumes
        self.masks = masks
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.force_fg_ratio = force_fg_ratio
        self.fg_min_ratio = fg_min_ratio
        self.use_roi = use_roi
        self.roi_margin = roi_margin
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return self.samples_per_volume * len(self.volumes)

    def __iter__(self):
        for vol_idx in range(len(self.volumes)):
            volume = self.volumes[vol_idx]
            mask = self.masks[vol_idx]
            roi_bounds = get_roi_bounds(mask, self.roi_margin) if self.use_roi else None
            num_fg_patches = int(self.samples_per_volume * self.force_fg_ratio)

            for i in range(self.samples_per_volume):
                force_fg = i < num_fg_patches
                patch_img, patch_mask, _ = get_random_patch(
                    volume, mask, self.patch_size,
                    force_fg=force_fg, fg_min_ratio=self.fg_min_ratio,
                    roi_bounds=roi_bounds,
                )

                img_tensor = torch.from_numpy(patch_img.astype(np.float32)).unsqueeze(0)
                seg_tensor = torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0)

                if self.augment_fn is not None:
                    img_tensor, seg_tensor = self.augment_fn(img_tensor, seg_tensor)

                yield img_tensor, seg_tensor.long().squeeze(0)
