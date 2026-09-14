"""Sliding-window 3D inference with Gaussian importance weighting."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm


class SlidingWindowPredictor:
    """Splits a volume into overlapping patches and aggregates with Gaussian weights."""

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: Tuple[int, int, int],
        num_classes: int,
        overlap: float = 0.5,
        batch_size: int = 4,
        device: str = "cuda",
        gaussian_sigma: float = 0.125,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device
        self.gaussian_sigma = gaussian_sigma

    def _gaussian_kernel(self, size: Tuple[int, int, int]) -> torch.Tensor:
        d, h, w = size
        sigma_d = d * self.gaussian_sigma
        sigma_h = h * self.gaussian_sigma
        sigma_w = w * self.gaussian_sigma

        d_coords = torch.arange(d, dtype=torch.float32, device=self.device) - (d - 1) / 2
        h_coords = torch.arange(h, dtype=torch.float32, device=self.device) - (h - 1) / 2
        w_coords = torch.arange(w, dtype=torch.float32, device=self.device) - (w - 1) / 2
        dd, hh, ww = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")

        kernel = torch.exp(
            -0.5 * ((dd / max(sigma_d, 1e-3)) ** 2
                    + (hh / max(sigma_h, 1e-3)) ** 2
                    + (ww / max(sigma_w, 1e-3)) ** 2)
        )
        return kernel / kernel.max()

    def _compute_window_positions(self, volume_shape: Tuple[int, int, int]) -> list:
        steps = [max(1, int(p * (1.0 - self.overlap))) for p in self.patch_size]
        positions = []
        for d in range(0, volume_shape[0], steps[0]):
            for h in range(0, volume_shape[1], steps[1]):
                for w in range(0, volume_shape[2], steps[2]):
                    d0 = min(d, max(0, volume_shape[0] - self.patch_size[0]))
                    h0 = min(h, max(0, volume_shape[1] - self.patch_size[1]))
                    w0 = min(w, max(0, volume_shape[2] - self.patch_size[2]))
                    positions.append((d0, h0, w0))
        return positions

    @torch.no_grad()
    def predict(self, volume: np.ndarray, return_logits: bool = False) -> np.ndarray:
        """Run sliding-window inference.

        Args:
            volume: (D, H, W) input image.
            return_logits: if True, return softmax probabilities (D,H,W,C).

        Returns:
            (D, H, W) integer labels, or (D, H, W, C) probabilities.
        """
        self.model.eval()
        original_shape = volume.shape

        pad_vol = any(s < p for s, p in zip(volume.shape, self.patch_size))
        if pad_vol:
            from ..dataset.preprocessing import pad_to_patch_size
            volume, _ = pad_to_patch_size(volume, self.patch_size)

        D, H, W = volume.shape
        pred_accum = np.zeros((D, H, W, self.num_classes), dtype=np.float32)
        weight_accum = np.zeros((D, H, W), dtype=np.float32)

        positions = self._compute_window_positions((D, H, W))
        gauss_kernel = self._gaussian_kernel(self.patch_size).cpu().numpy()

        for batch_start in tqdm(range(0, len(positions), self.batch_size),
                                desc="Sliding window"):
            batch_positions = positions[batch_start:batch_start + self.batch_size]

            batch_patches = []
            for d0, h0, w0 in batch_positions:
                patch = volume[d0:d0 + self.patch_size[0],
                               h0:h0 + self.patch_size[1],
                               w0:w0 + self.patch_size[2]]
                if patch.shape != self.patch_size:
                    padded = np.zeros(self.patch_size, dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                batch_patches.append(patch)

            batch_tensor = torch.from_numpy(
                np.stack([p[np.newaxis, ...] for p in batch_patches])
            ).to(self.device)

            outputs = self.model(batch_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # main logits during deep-supervision training
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            for i, (d0, h0, w0) in enumerate(batch_positions):
                act_d = min(self.patch_size[0], D - d0)
                act_h = min(self.patch_size[1], H - h0)
                act_w = min(self.patch_size[2], W - w0)
                kernel = gauss_kernel[:act_d, :act_h, :act_w]

                for c in range(self.num_classes):
                    pred_accum[d0:d0 + act_d, h0:h0 + act_h, w0:w0 + act_w, c] += (
                        probs[i, c, :act_d, :act_h, :act_w] * kernel
                    )
                weight_accum[d0:d0 + act_d, h0:h0 + act_h, w0:w0 + act_w] += kernel

        weight_accum = np.maximum(weight_accum, 1e-6)
        for c in range(self.num_classes):
            pred_accum[..., c] /= weight_accum

        if pad_vol:
            pred_accum = pred_accum[:original_shape[0], :original_shape[1], :original_shape[2]]

        if return_logits:
            return pred_accum
        return np.argmax(pred_accum, axis=-1).astype(np.uint8)


def connected_component_postprocessing(
    segmentation: np.ndarray,
    min_volume_mm3: float = 100.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    keep_largest_only: bool = False,
) -> np.ndarray:
    """Remove small connected components (false-positive islands)."""
    from scipy.ndimage import label as connected_label

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    min_voxels = max(1, int(min_volume_mm3 / voxel_volume_mm3))

    cleaned = np.zeros_like(segmentation)
    for class_val in np.unique(segmentation):
        if class_val == 0:
            continue
        binary = (segmentation == class_val)
        labeled, num_features = connected_label(binary)
        if keep_largest_only and num_features > 1:
            sizes = [(labeled == c).sum() for c in range(1, num_features + 1)]
            largest = int(np.argmax(sizes)) + 1
            cleaned[labeled == largest] = class_val
            continue
        for comp_id in range(1, num_features + 1):
            component = (labeled == comp_id)
            if component.sum() >= min_voxels:
                cleaned[component] = class_val

    return cleaned
