import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm


class SlidingWindowPredictor:
    """
    Sliding-window 3D predictor with Gaussian importance weighting.

    Splits the volume into overlapping patches, runs inference on each,
    and aggregates results with a Gaussian kernel that gives higher weight
    to voxels near the center of each patch.
    """

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
        """
        Args:
            model: Trained nnU-Net model.
            patch_size: (D, H, W) inference patch size (same as training).
            num_classes: Number of output classes.
            overlap: Overlap fraction between adjacent windows (0.0–1.0).
            batch_size: Number of patches to process simultaneously on GPU.
            device: "cuda" or "cpu".
            gaussian_sigma: Sigma for Gaussian weighting kernel (fraction of patch size).
        """
        self.model = model.to(device)
        self.model.eval()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = device
        self.gaussian_sigma = gaussian_sigma

    def _gaussian_kernel(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Create a 3D Gaussian kernel on the GPU."""
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

    def _compute_window_positions(
        self, volume_shape: Tuple[int, int, int]
    ) -> list:
        """Compute all window start positions with specified overlap."""
        steps = [
            max(1, int(p * (1.0 - self.overlap)))
            for p in self.patch_size
        ]
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
    def predict(
        self,
        volume: np.ndarray,
        return_logits: bool = False,
    ) -> np.ndarray:
        """
        Run sliding-window inference on a 3D volume.

        Args:
            volume: (D, H, W) input image (numpy).
            return_logits: If True, return softmax probabilities instead of labels.

        Returns:
            (D, H, W) integer segmentation or (D, H, W, C) probabilities.
        """
        self.model.eval()
        D, H, W = volume.shape

        # Ensure volume is large enough; if not, pad
        pad_vol = False
        if D < self.patch_size[0] or H < self.patch_size[1] or W < self.patch_size[2]:
            pad_vol = True
            from ..dataset.preprocessing import pad_to_patch_size
            volume, original_shape = pad_to_patch_size(volume, self.patch_size)
            D, H, W = volume.shape

        # Output accumulators
        pred_accum = np.zeros((D, H, W, self.num_classes), dtype=np.float32)
        weight_accum = np.zeros((D, H, W), dtype=np.float32)

        positions = self._compute_window_positions((D, H, W))

        # Create Gaussian kernel once
        gauss_kernel = self._gaussian_kernel(self.patch_size).cpu().numpy()

        # Process patches in batches
        for batch_start in tqdm(range(0, len(positions), self.batch_size),
                                desc="Sliding window"):
            batch_positions = positions[batch_start:batch_start + self.batch_size]

            # Prepare batch
            batch_patches = []
            for d0, h0, w0 in batch_positions:
                d_end = d0 + self.patch_size[0]
                h_end = h0 + self.patch_size[1]
                w_end = w0 + self.patch_size[2]
                patch = volume[d0:d_end, h0:h_end, w0:w_end]
                # Handling last window
                if patch.shape != self.patch_size:
                    pad_patch = np.zeros(self.patch_size, dtype=np.float32)
                    pad_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = pad_patch
                batch_patches.append(patch)

            # (B, 1, D, H, W)
            batch_tensor = torch.from_numpy(
                np.stack([p[np.newaxis, ...] for p in batch_patches])
            ).to(self.device)

            # Inference
            outputs = self.model(batch_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # main logits during deep supervision training
            probs = F.softmax(outputs, dim=1)  # (B, C, D, H, W)
            probs_np = probs.cpu().numpy()

            # Aggregate
            for i, (d0, h0, w0) in enumerate(batch_positions):
                d_end = d0 + self.patch_size[0]
                h_end = h0 + self.patch_size[1]
                w_end = w0 + self.patch_size[2]

                # Crop if last window
                act_d = min(self.patch_size[0], D - d0)
                act_h = min(self.patch_size[1], H - h0)
                act_w = min(self.patch_size[2], W - w0)

                for c in range(self.num_classes):
                    pred_accum[d0:d0 + act_d, h0:h0 + act_h, w0:w0 + act_w, c] += (
                        probs_np[i, c, :act_d, :act_h, :act_w] * gauss_kernel[:act_d, :act_h, :act_w]
                    )
                weight_accum[d0:d0 + act_d, h0:h0 + act_h, w0:w0 + act_w] += (
                    gauss_kernel[:act_d, :act_h, :act_w]
                )

        # Normalize by weights
        weight_accum = np.maximum(weight_accum, 1e-6)
        for c in range(self.num_classes):
            pred_accum[..., c] /= weight_accum

        # Unpad if needed
        if pad_vol:
            pred_accum = pred_accum[:original_shape[0], :original_shape[1], :original_shape[2]]
            # Fake original_shape variable — we need to recompute it
            od, oh, ow = volume.shape  # get from padded
            # Actually we recompute: crop back to original

        if return_logits:
            return pred_accum
        else:
            return np.argmax(pred_accum, axis=-1).astype(np.uint8)


def connected_component_postprocessing(
    segmentation: np.ndarray,
    min_volume_mm3: float = 100.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Remove small connected components from the segmentation.
    Useful for removing false positive islands in pancreas segmentation.

    Args:
        segmentation: (D, H, W) integer label map.
        min_volume_mm3: Minimum volume in mm³ to keep.
        spacing: Voxel spacing (mm).

    Returns:
        Cleaned segmentation.
    """
    from scipy.ndimage import label as connected_label

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    min_voxels = max(1, int(min_volume_mm3 / voxel_volume_mm3))

    cleaned = np.zeros_like(segmentation)
    for class_val in np.unique(segmentation):
        if class_val == 0:
            continue
        binary = (segmentation == class_val)
        labeled, num_features = connected_label(binary)
        for comp_id in range(1, num_features + 1):
            component = (labeled == comp_id)
            if component.sum() >= min_voxels:
                cleaned[component] = class_val

    return cleaned
