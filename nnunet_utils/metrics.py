"""Segmentation metrics for final evaluation: Dice, Recall, HD95, ASSD, clDice, bbox IoU.

nnU-Net reports Dice internally during training but does not ship a standalone
Recall / HD95 / ASSD / clDice command, so these are computed here against a
ground-truth folder (see ``evaluate.py``). Surface distances (HD95 / ASSD) accept
the voxel ``spacing`` so they are reported in millimetres, not voxels.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize_3d


def dice_score(pred, gt):
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * float(inter) / float(denom)


def recall(pred, gt):
    """Recall / sensitivity = |pred ∩ gt| / |gt|."""
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    gt_sum = gt.sum()
    if gt_sum == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum()) / float(gt_sum)


def hausdorff_distance_95(pred, gt, spacing=(1, 1, 1)):
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    p_surf = pred ^ binary_erosion(pred)
    g_surf = gt ^ binary_erosion(gt)
    dp = distance_transform_edt(~p_surf, sampling=spacing)
    dg = distance_transform_edt(~g_surf, sampling=spacing)
    return float(max(np.percentile(dp[g_surf], 95), np.percentile(dg[p_surf], 95)))


def assd(pred, gt, spacing=(1, 1, 1)):
    """Average symmetric surface distance (in mm, given the voxel spacing)."""
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    p_surf = pred ^ binary_erosion(pred)
    g_surf = gt ^ binary_erosion(gt)
    dp = distance_transform_edt(~p_surf, sampling=spacing)
    dg = distance_transform_edt(~g_surf, sampling=spacing)
    n = int(g_surf.sum() + p_surf.sum())
    if n == 0:
        return float("nan")
    return float((dp[g_surf].sum() + dg[p_surf].sum()) / n)


def cl_dice(pred, gt):
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    skel_pred = skeletonize_3d(pred)
    skel_gt = skeletonize_3d(gt)
    tprec = skel_pred[gt].sum() / max(skel_pred.sum(), 1e-12)
    tsens = skel_gt[pred].sum() / max(skel_gt.sum(), 1e-12)
    if tprec + tsens == 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


def bbox_iou(pred, gt):
    """IoU of the tight 3D bounding boxes of two binary masks (ROI localization)."""
    pz, py, px = np.nonzero(np.asarray(pred) > 0.5)
    gz, gy, gx = np.nonzero(np.asarray(gt) > 0.5)
    if len(pz) == 0 or len(gz) == 0:
        return float("nan")
    p = (pz.min(), py.min(), px.min(), pz.max(), py.max(), px.max())
    g = (gz.min(), gy.min(), gx.min(), gz.max(), gy.max(), gx.max())

    inter = [max(p[0], g[0]), max(p[1], g[1]), max(p[2], g[2]),
             min(p[3], g[3]), min(p[4], g[4]), min(p[5], g[5])]
    iv = (inter[3] - inter[0] + 1) * (inter[4] - inter[1] + 1) * (inter[5] - inter[2] + 1)
    if iv <= 0:
        return 0.0
    pv = (p[3] - p[0] + 1) * (p[4] - p[1] + 1) * (p[5] - p[2] + 1)
    gv = (g[3] - g[0] + 1) * (g[4] - g[1] + 1) * (g[5] - g[2] + 1)
    return float(iv) / max(pv + gv - iv, 1)
