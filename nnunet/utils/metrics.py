"""Segmentation metrics: Dice, Hausdorff distance, clDice."""

import numpy as np
import torch
from typing import List, Optional, Dict


def dice_score(pred, target, smooth: float = 1e-5) -> float:
    """Mean Dice over foreground classes (class 1..C-1)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    num_classes = int(max(pred.max(), target.max())) + 1
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom == 0:
            dices.append(1.0)
        else:
            dices.append(2.0 * intersection / (denom + smooth))
    return float(np.mean(dices)) if dices else 0.0


def dice_per_class(pred, target, num_classes: int, smooth: float = 1e-5) -> List[float]:
    """Dice score for each class individually."""
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
        dices.append(1.0 if denom == 0 else 2.0 * intersection / (denom + smooth))
    return dices


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray,
                          spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance (mm) between two binary masks."""
    from scipy.ndimage import binary_erosion

    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    pred_surface = pred & ~binary_erosion(pred)
    target_surface = target & ~binary_erosion(target)

    if not pred_surface.any() or not target_surface.any():
        return float("nan")

    pred_pts = np.argwhere(pred_surface).astype(np.float64)
    target_pts = np.argwhere(target_surface).astype(np.float64)
    spacing_arr = np.array(spacing, dtype=np.float64).reshape(1, -1)
    pred_pts *= spacing_arr
    target_pts *= spacing_arr

    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pred_pts)
    tree_target = cKDTree(target_pts)
    dist_p2t, _ = tree_target.query(pred_pts)
    dist_t2p, _ = tree_pred.query(target_pts)

    return float(np.percentile(np.concatenate([dist_p2t, dist_t2p]), 95))


def cl_dice(pred, target, smooth: float = 1e-5) -> float:
    """Centerline Dice (clDice) between two binary masks (Experiment 5).

    Uses 3D morphological skeletonization; rewards topological continuity of
    thin/tubular structures regardless of local thickness mismatch.
    """
    from skimage.morphology import skeletonize_3d

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p = (np.asarray(pred) > 0).astype(np.uint8)
    t = (np.asarray(target) > 0).astype(np.uint8)

    if p.sum() == 0 or t.sum() == 0:
        return 0.0

    skel_p = skeletonize_3d(p).astype(np.float32)
    skel_t = skeletonize_3d(t).astype(np.float32)

    tprec = (p * skel_t).sum() / (p.sum() + smooth)
    tsens = (t * skel_p).sum() / (skel_p.sum() + smooth)

    return float(2.0 * tprec * tsens / (tprec + tsens + smooth))


def compute_metrics(pred, target, spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """Compute a standard metric summary for one case."""
    return {
        "dice": dice_score(pred, target),
        "hd95": hausdorff_distance_95(np.asarray(pred) > 0, np.asarray(target) > 0, spacing),
        "cl_dice": cl_dice(pred, target),
    }


def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of per-case metric dicts (skipping NaN values)."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None and not np.isnan(m.get(k, float("nan")))]
        summary[k] = float(np.mean(vals)) if vals else float("nan")
    return summary
