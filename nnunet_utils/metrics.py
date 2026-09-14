"""Segmentation metrics for final evaluation: Dice, HD95, clDice.

nnU-Net reports Dice internally during training but does not ship a standalone
HD95 / clDice command, so these are computed here against a ground-truth folder
(see ``evaluate.py``).
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


def hausdorff_distance_95(pred, gt):
    pred = np.asarray(pred) > 0.5
    gt = np.asarray(gt) > 0.5
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    p_surf = pred ^ binary_erosion(pred)
    g_surf = gt ^ binary_erosion(gt)
    dp = distance_transform_edt(~p_surf)
    dg = distance_transform_edt(~g_surf)
    return float(max(np.percentile(dp[g_surf], 95), np.percentile(dg[p_surf], 95)))


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
