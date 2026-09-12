import numpy as np
import torch
from typing import List, Optional
from scipy.ndimage import binary_erosion


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Dice score (hard labels).
    pred, target: (H, W, D) or (C, H, W, D) numpy/torch arrays of integer labels.
    Returns mean Dice over foreground classes (class 1..C-1).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    num_classes = int(max(pred.max(), target.max())) + 1
    dices = []
    for c in range(1, num_classes):  # skip background
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom == 0:
            dices.append(1.0)  # both empty → perfect agreement
        else:
            dices.append(2.0 * intersection / (denom + smooth))
    return float(np.mean(dices)) if dices else 0.0


def dice_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-5
) -> List[float]:
    """Return Dice score for each class."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    dices = []
    for c in range(num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom == 0:
            dices.append(1.0)
        else:
            dices.append(2.0 * intersection / (denom + smooth))
    return dices


def hausdorff_distance_95(
    pred: np.ndarray, target: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute the 95th percentile Hausdorff distance (in mm).
    Uses SimpleITK if available for surface calculation.
    """
    # Surface extraction via binary erosion
    pred_surface = pred.astype(bool) & ~binary_erosion(pred.astype(bool))
    target_surface = target.astype(bool) & ~binary_erosion(target.astype(bool))

    if not pred_surface.any() or not target_surface.any():
        return float("nan")

    pred_pts = np.argwhere(pred_surface).astype(np.float64)
    target_pts = np.argwhere(target_surface).astype(np.float64)
    spacing_arr = np.array(spacing).reshape(1, -1)
    pred_pts *= spacing_arr
    target_pts *= spacing_arr

    from scipy.spatial import cKDTree

    tree_pred = cKDTree(pred_pts)
    tree_target = cKDTree(target_pts)

    dist_p2t, _ = tree_target.query(pred_pts)
    dist_t2p, _ = tree_pred.query(target_pts)

    all_dists = np.concatenate([dist_p2t, dist_t2p])
    return float(np.percentile(all_dists, 95))
