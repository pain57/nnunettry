#!/usr/bin/env python3
"""Evaluate predictions against ground truth, per foreground class.

Metrics: Dice, Recall, HD95 (mm), ASSD (mm), clDice. Surface distances use the
ground-truth voxel spacing so HD95 / ASSD are reported in millimetres.

    python evaluate.py --pred_dir predictions \
        --gt_dir nnUNet_raw/Dataset150_PancreasDuct/labelsTs

ROI localization check (coarse stage only, bbox IoU):
    python evaluate.py --pred_dir <roi_preds> --gt_dir ... --localization
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet_utils.metrics import (
    assd, bbox_iou, cl_dice, dice_score, hausdorff_distance_95, recall,
)

FOREGROUND = {1: "pancreatic_duct", 2: "bile_duct"}
METRICS = ("Dice", "Recall", "HD95(mm)", "ASSD(mm)", "clDice")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--localization", action="store_true",
                    help="report bbox IoU (coarse ROI localization) instead of segmentation metrics")
    args = ap.parse_args()

    pred_files = _case_files(args.pred_dir)

    if args.localization:
        ious = []
        for pf in pred_files:
            case_id = pf.name[:-7] if pf.name.endswith(".nii.gz") else pf.name[:-4]
            gt = _find_gt(args.gt_dir, case_id)
            if gt is None:
                print(f"[skip] no ground truth for {case_id}")
                continue
            iou = bbox_iou(nib.load(pf).get_fdata() > 0, nib.load(gt).get_fdata() > 0)
            ious.append(iou)
            print(f"{case_id}:  bbox IoU={iou:.4f}")
        if ious:
            print("\nMean bbox IoU = %.4f" % np.nanmean(ious))
        return

    per_class = {c: [] for c in FOREGROUND}
    for pf in pred_files:
        case_id = pf.name[:-7] if pf.name.endswith(".nii.gz") else pf.name[:-4]
        gt = _find_gt(args.gt_dir, case_id)
        if gt is None:
            print(f"[skip] no ground truth for {case_id}")
            continue

        pred = nib.load(pf).get_fdata().astype(np.uint8)
        gt_nii = nib.load(gt)
        gt_data = gt_nii.get_fdata().astype(np.uint8)
        spacing = _spacing(gt_nii)

        parts = []
        for c, name in FOREGROUND.items():
            p, g = pred == c, gt_data == c
            row = (dice_score(p, g), recall(p, g),
                   hausdorff_distance_95(p, g, spacing=spacing),
                   assd(p, g, spacing=spacing), cl_dice(p, g))
            per_class[c].append(row)
            parts.append(f"{name[:3].upper()}({', '.join('%.3f' % v for v in row)})")
        print(f"{case_id}:  " + "  ".join(parts))

    print(f"\nMean per class ({', '.join(METRICS)}):")
    for c, name in FOREGROUND.items():
        arr = np.array(per_class[c], dtype=float)
        if arr.size == 0:
            continue
        mean = np.nanmean(arr, axis=0)
        print(f"  {name:15s}  " + "  ".join("%.4f" % v for v in mean))


if __name__ == "__main__":
    main()
