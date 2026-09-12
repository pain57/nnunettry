import torch
import numpy as np
from typing import Tuple, Optional, List


def get_random_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    force_fg: bool = False,
    fg_min_ratio: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Sample a random patch from a 3D volume.

    Args:
        volume: (D, H, W) image volume.
        mask: (D, H, W) label mask.
        patch_size: (PD, PH, PW) desired patch shape.
        force_fg: If True, resample until patch contains at least fg_min_ratio foreground.
        fg_min_ratio: Minimum fraction of foreground voxels in the patch.

    Returns:
        (image_patch, mask_patch, start_coords)
    """
    D, H, W = volume.shape
    pD, pH, pW = patch_size

    if force_fg and mask.sum() > 0:
        # Try up to 50 times to get a patch with foreground
        for _ in range(50):
            d0 = np.random.randint(0, max(1, D - pD))
            h0 = np.random.randint(0, max(1, H - pH))
            w0 = np.random.randint(0, max(1, W - pW))
            d_end = min(d0 + pD, D)
            h_end = min(h0 + pH, H)
            w_end = min(w0 + pW, W)
            patch_mask = mask[d0:d_end, h0:h_end, w0:w_end]
            fg_ratio = patch_mask.sum() / patch_mask.size
            if fg_ratio >= fg_min_ratio:
                patch_img = volume[d0:d_end, h0:h_end, w0:w_end]
                return patch_img, patch_mask, (d0, h0, w0)

    # Fallback: random patch
    d0 = np.random.randint(0, max(1, D - pD)) if D > pD else 0
    h0 = np.random.randint(0, max(1, H - pH)) if H > pH else 0
    w0 = np.random.randint(0, max(1, W - pW)) if W > pW else 0
    d_end = min(d0 + pD, D)
    h_end = min(h0 + pH, H)
    w_end = min(w0 + pW, W)
    patch_img = volume[d0:d_end, h0:h_end, w0:w_end]
    patch_mask = mask[d0:d_end, h0:h_end, w0:w_end]
    return patch_img, patch_mask, (d0, h0, w0)


class PatchSampler:
    """
    Patch-based 3D data sampler.

    Yields (image_patch, mask_patch) pairs from a list of volumes,
    with class-balanced foreground sampling.
    """

    def __init__(
        self,
        volumes: List[np.ndarray],
        masks: List[np.ndarray],
        patch_size: Tuple[int, int, int],
        samples_per_volume: int = 8,
        force_fg_ratio: float = 0.33,
        fg_min_ratio: float = 0.01,
        fg_class_value: int = 1,
    ):
        """
        Args:
            volumes: List of 3D image arrays.
            masks: List of 3D label arrays.
            patch_size: (D, H, W) patch dimensions.
            samples_per_volume: Number of patches to draw per volume per epoch.
            force_fg_ratio: Fraction of patches that must contain foreground.
            fg_min_ratio: Minimum foreground ratio threshold.
        """
        self.volumes = volumes
        self.masks = masks
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.force_fg_ratio = force_fg_ratio
        self.fg_min_ratio = fg_min_ratio

        self.num_samples = samples_per_volume * len(volumes)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        return self._generate()

    def _generate(self):
        for vol_idx in range(len(self.volumes)):
            volume = self.volumes[vol_idx]
            mask = self.masks[vol_idx]
            num_fg_patches = int(self.samples_per_volume * self.force_fg_ratio)

            for i in range(self.samples_per_volume):
                force_fg = i < num_fg_patches
                patch_img, patch_mask, _ = get_random_patch(
                    volume, mask, self.patch_size,
                    force_fg=force_fg, fg_min_ratio=self.fg_min_ratio,
                )
                yield patch_img, patch_mask


class PatchDataset3D(torch.utils.data.IterableDataset):
    """
    Torch IterableDataset for patch-based 3D training.

    Args:
        volumes: List of preprocessed image volumes (D, H, W) numpy arrays.
        masks: List of preprocessed label masks (D, H, W) numpy arrays.
        patch_size: (D, H, W) patch dimensions.
        samples_per_volume: Patches per volume per epoch.
        force_fg_ratio: Fraction of patches forced to contain foreground.
        fg_min_ratio: Min foreground ratio to count as "contains fg".
        augment_fn: Optional augmentation function (image, label) -> (image, label).
    """

    def __init__(
        self,
        volumes: List[np.ndarray],
        masks: List[np.ndarray],
        patch_size: Tuple[int, int, int],
        samples_per_volume: int = 8,
        force_fg_ratio: float = 0.33,
        fg_min_ratio: float = 0.01,
        augment_fn=None,
    ):
        super().__init__()
        self.volumes = volumes
        self.masks = masks
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.force_fg_ratio = force_fg_ratio
        self.fg_min_ratio = fg_min_ratio
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return self.samples_per_volume * len(self.volumes)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Simple single-worker iteration
        for vol_idx in range(len(self.volumes)):
            volume = self.volumes[vol_idx]
            mask = self.masks[vol_idx]
            num_fg_patches = int(self.samples_per_volume * self.force_fg_ratio)

            for i in range(self.samples_per_volume):
                force_fg = i < num_fg_patches
                patch_img, patch_mask, _ = get_random_patch(
                    volume, mask, self.patch_size,
                    force_fg=force_fg, fg_min_ratio=self.fg_min_ratio,
                )

                # Convert to torch tensors: (1, D, H, W) and (1, D, H, W)
                img_tensor = torch.from_numpy(patch_img.astype(np.float32)).unsqueeze(0)
                seg_tensor = torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0)

                # Apply augmentation on GPU-compatible tensors
                if self.augment_fn is not None:
                    img_tensor, seg_tensor = self.augment_fn(img_tensor, seg_tensor)

                yield img_tensor, seg_tensor.long().squeeze(0)
