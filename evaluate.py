#!/usr/bin/env python3
"""Evaluate predictions against ground truth: Dice / HD95 / clDice.

    python evaluate.py --pred_dir predictions --gt_dir nnUNet_raw/Dataset150_PancreasCT/labelsTs
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet_utils.metrics import cl_dice, dice_score, hausdorff_distance_95


def _case_files(d, suffix=".nii.gz"):
    files = sorted(Path(d).glob(f"*{suffix}"))
    if not files:
        files = sorted(Path(d).glob("*.nii"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    args = ap.parse_args()

    pred_files = _case_files(args.pred_dir)
    results = []
    for pf in pred_files:
        case_id = pf.name[:-7] if pf.name.endswith(".nii.gz") else pf.name[:-4]
        gt = Path(args.gt_dir) / f"{case_id}.nii.gz"
        if not gt.exists():
            gt = Path(args.gt_dir) / f"{case_id}.nii"
        if not gt.exists():
            print(f"[skip] no ground truth for {case_id}")
            continue

        pred = (nib.load(pf).get_fdata() > 0).astype(np.uint8)
        gt_data = (nib.load(gt).get_fdata() > 0).astype(np.uint8)
        d = dice_score(pred, gt_data)
        h = hausdorff_distance_95(pred, gt_data)
        c = cl_dice(pred, gt_data)
        results.append((case_id, d, h, c))
        print(f"{case_id}:  Dice={d:.4f}  HD95={h:.2f}  clDice={c:.4f}")

    if results:
        arr = np.array([r[1:] for r in results], dtype=float)
        mean = np.nanmean(arr, axis=0)
        print("\nMean  Dice=%.4f  HD95=%.2f  clDice=%.4f" % tuple(mean))


if __name__ == "__main__":
    main()
