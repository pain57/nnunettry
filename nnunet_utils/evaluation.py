"""Folder-level evaluation shared by ``evaluate.py`` and ``compare_experiments.py``.

Metrics are computed per foreground class (pancreas); HD95 and ASSD use the
ground-truth voxel spacing so they are reported in millimetres.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from .metrics import assd, bbox_iou, dice_score, hausdorff_distance_95, precision, recall

FOREGROUND = {1: "pancreas"}
METRIC_NAMES = ("Dice", "Recall", "Precision", "HD95(mm)", "ASSD(mm)")


def _case_files(d, suffix=".nii.gz"):
    files = sorted(Path(d).glob(f"*{suffix}"))
    if not files:
        files = sorted(Path(d).glob("*.nii"))
    return files


def _find_gt(gt_dir, case_id):
    gt = Path(gt_dir) / f"{case_id}.nii.gz"
    if gt.exists():
        return gt
    gt = Path(gt_dir) / f"{case_id}.nii"
    return gt if gt.exists() else None


def _spacing(nii):
    return tuple(float(s) for s in nii.header.get_zooms()[:3])  # (z, y, x)


def _case_id(name):
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def evaluate_folder(pred_dir, gt_dir):
    """Evaluate one prediction folder against a GT folder.

    Returns ``{"cases": {case_id: {class: [Dice, Recall, Precision, HD95, ASSD]}},
    "means": {class: [mean over cases]}}``.
    """
    cases = {}
    for pf in _case_files(pred_dir):
        case_id = _case_id(pf.name)
        gt = _find_gt(gt_dir, case_id)
        if gt is None:
            continue

        pred = nib.load(pf).get_fdata().astype(np.uint8)
        gt_nii = nib.load(gt)
        gt_data = gt_nii.get_fdata().astype(np.uint8)
        spacing = _spacing(gt_nii)

        row = {}
        for c in FOREGROUND:
            p, g = pred == c, gt_data == c
            row[c] = [dice_score(p, g), recall(p, g), precision(p, g),
                      hausdorff_distance_95(p, g, spacing=spacing),
                      assd(p, g, spacing=spacing)]
        cases[case_id] = row

    means = {}
    for c in FOREGROUND:
        arr = np.array([cases[cid][c] for cid in cases], dtype=float) if cases else np.empty((0, len(METRIC_NAMES)))
        means[c] = np.nanmean(arr, axis=0) if arr.size else np.full(len(METRIC_NAMES), np.nan)
    return {"cases": cases, "means": means}


def evaluate_localization(pred_dir, gt_dir):
    """bbox IoU per case (quick localization check, not a segmentation metric)."""
    cases = {}
    for pf in _case_files(pred_dir):
        case_id = _case_id(pf.name)
        gt = _find_gt(gt_dir, case_id)
        if gt is None:
            continue
        cases[case_id] = bbox_iou(nib.load(pf).get_fdata() > 0, nib.load(gt).get_fdata() > 0)
    ious = list(cases.values())
    return {"cases": cases, "mean": float(np.nanmean(ious)) if ious else float("nan")}
