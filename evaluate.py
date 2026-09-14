#!/usr/bin/env python3
"""Evaluate segmentations against ground-truth labels.

Computes per-case Dice, 95th-percentile Hausdorff distance and clDice, then
prints mean summaries:

    python evaluate.py --pred_dir predictions --gt_dir data/labelsTs
"""

import argparse
from pathlib import Path

import numpy as np

from nnunet.utils.nifti_io import load_nifti_with_spacing
from nnunet.utils.metrics import compute_metrics, summarize_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate pancreas segmentations")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory of predictions (*_seg.nii.gz)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of ground-truth labels")
    parser.add_argument("--spacing", type=str, default=None,
                        help="Override voxel spacing as 'a,b,c' for HD95 (default: from GT header)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = sorted(pred_dir.glob("*_seg.nii.gz"))
    if not pred_files:
        # fall back to any nii.gz
        pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"No predictions found in {pred_dir}")
        return

    metrics_list = []
    for pred_file in pred_files:
        # Match ground truth by stripping the '_seg' suffix.
        case_name = pred_file.name.replace("_seg.nii.gz", "").replace(".nii.gz", "")
        gt_file = gt_dir / f"{case_name}.nii.gz"
        if not gt_file.exists():
            gt_file = gt_dir / pred_file.name.replace("_seg.nii.gz", ".nii.gz")
        if not gt_file.exists():
            print(f"  [skip] no ground truth for {pred_file.name}")
            continue

        pred, _, _ = load_nifti_with_spacing(str(pred_file))
        gt, _, spacing = load_nifti_with_spacing(str(gt_file))

        if args.spacing:
            spacing = tuple(float(x) for x in args.spacing.split(","))

        m = compute_metrics(pred, gt, spacing=spacing)
        metrics_list.append(m)
        print(f"  {case_name}: Dice={m['dice']:.4f} | HD95={m['hd95']:.2f}mm | clDice={m['cl_dice']:.4f}")

    if metrics_list:
        summary = summarize_metrics(metrics_list)
        print("\n=== Summary ===")
        print(f"  Mean Dice:  {summary['dice']:.4f}")
        print(f"  Mean HD95:  {summary['hd95']:.2f} mm")
        print(f"  Mean clDice: {summary['cl_dice']:.4f}")
    else:
        print("\nNo cases could be evaluated.")


if __name__ == "__main__":
    main()
