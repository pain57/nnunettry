"""Coarse-to-fine / ROI inference (Experiment 4).

Two-stage segmentation for extreme class imbalance: a coarse model localizes the
pancreas in the full volume, then a fine model segments only the cropped region
of interest around the coarse prediction. This lets the fine model operate at a
higher effective resolution with a much larger foreground ratio.
"""

import numpy as np
from typing import Tuple, Optional
import torch

from ..dataset.patch_sampler import get_roi_bounds
from .predictor import SlidingWindowPredictor


class CoarseToFinePredictor:
    """Runs a coarse model then a fine model restricted to the predicted ROI."""

    def __init__(
        self,
        coarse_model: torch.nn.Module,
        fine_model: torch.nn.Module,
        coarse_patch_size: Tuple[int, int, int],
        fine_patch_size: Tuple[int, int, int],
        num_classes: int,
        roi_margin: int = 16,
        overlap: float = 0.5,
        batch_size: int = 4,
        device: str = "cuda",
        gaussian_sigma: float = 0.125,
    ):
        self.coarse_predictor = SlidingWindowPredictor(
            model=coarse_model, patch_size=coarse_patch_size, num_classes=num_classes,
            overlap=overlap, batch_size=batch_size, device=device,
            gaussian_sigma=gaussian_sigma,
        )
        self.fine_predictor = SlidingWindowPredictor(
            model=fine_model, patch_size=fine_patch_size, num_classes=num_classes,
            overlap=overlap, batch_size=batch_size, device=device,
            gaussian_sigma=gaussian_sigma,
        )
        self.roi_margin = roi_margin

    def predict(self, volume: np.ndarray, return_roi: bool = False):
        """Run coarse-to-fine inference.

        Args:
            volume: (D, H, W) input image.
            return_roi: if True, also return the ROI slices.

        Returns:
            (D, H, W) integer labels (same shape as input), and optionally the
            ROI bounding box as ((z0,z1),(y0,y1),(x0,x1)).
        """
        # Step 1: coarse localization on the full volume.
        coarse_pred = self.coarse_predictor.predict(volume)

        # Step 2: crop the ROI around the coarse prediction.
        bounds = get_roi_bounds(coarse_pred, self.roi_margin)
        if bounds is None:
            empty = np.zeros_like(volume, dtype=np.uint8)
            return (empty, bounds) if return_roi else empty

        (z0, z1), (y0, y1), (x0, x1) = bounds
        roi = volume[z0:z1, y0:y1, x0:x1]

        # Step 3: fine segmentation within the ROI.
        fine_pred = self.fine_predictor.predict(roi)

        # Step 4: paste the fine result back into the full-resolution label map.
        out = np.zeros_like(volume, dtype=np.uint8)
        out[z0:z1, y0:y1, x0:x1] = fine_pred

        return (out, bounds) if return_roi else out


def predict_coarse_to_fine(
    volume: np.ndarray,
    coarse_predictor: SlidingWindowPredictor,
    fine_predictor: SlidingWindowPredictor,
    roi_margin: int = 16,
) -> Tuple[np.ndarray, Optional[Tuple[Tuple[int, int], ...]]]:
    """Functional two-stage prediction helper.

    Returns (full_volume_segmentation, roi_bounds).
    """
    coarse_pred = coarse_predictor.predict(volume)
    bounds = get_roi_bounds(coarse_pred, roi_margin)
    if bounds is None:
        return np.zeros_like(volume, dtype=np.uint8), bounds

    (z0, z1), (y0, y1), (x0, x1) = bounds
    roi = volume[z0:z1, y0:y1, x0:x1]
    fine_pred = fine_predictor.predict(roi)

    out = np.zeros_like(volume, dtype=np.uint8)
    out[z0:z1, y0:y1, x0:x1] = fine_pred
    return out, bounds
