#!/usr/bin/env python3
"""Evaluate predictions against ground truth.

Per-class Dice / HD95 / clDice (default):
    python evaluate.py --pred_dir predictions \
        --gt_dir nnUNet_raw/Dataset150_PancreasDuct/labelsTs

ROI localization (Exp 4, bbox IoU on the union of all foreground):
    python evaluate.py --pred_dir <lowres_preds> --gt_dir ... --localization
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet_utils.metrics import bbox_iou, cl_dice, dice_score, hausdorff_distance_95

FOREGROUND = {1: "pancreatic_duct", 2: "bile_duct"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--localization", action="store_true",
                    help="Exp 4: report bbox IoU (ROI localization) instead of segmentation metrics")
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
            pred = nib.load(pf).get_fdata() > 0
            gt_data = nib.load(gt).get_fdata() > 0
            iou = bbox_iou(pred, gt_data)
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
        gt_data = nib.load(gt).get_fdata().astype(np.uint8)

        parts = []
        for c, name in FOREGROUND.items():
            d = dice_score(pred == c, gt_data == c)
            h = hausdorff_distance_95(pred == c, gt_data == c)
            cl = cl_dice(pred == c, gt_data == c)
            per_class[c].append((d, h, cl))
            parts.append(f"{name[:3].upper()} Dice={d:.4f} HD95={h:.2f} clDice={cl:.4f}")
        print(f"{case_id}:  " + "  ".join(parts))

    print("\nMean per class:")
    for c, name in FOREGROUND.items():
        arr = np.array(per_class[c], dtype=float)
        if arr.size == 0:
            continue
        mean = np.nanmean(arr, axis=0)
        print(f"  {name:15s}  Dice=%.4f  HD95=%.2f  clDice=%.4f" % tuple(mean))


if __name__ == "__main__":
    main()
