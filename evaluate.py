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

from nnunet_utils.evaluation import (
    FOREGROUND, METRIC_NAMES, evaluate_folder, evaluate_localization,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--localization", action="store_true",
                    help="report bbox IoU (coarse ROI localization) instead of segmentation metrics")
    args = ap.parse_args()

    if args.localization:
        res = evaluate_localization(args.pred_dir, args.gt_dir)
        for case_id, iou in res["cases"].items():
            print(f"{case_id}:  bbox IoU={iou:.4f}")
        print("\nMean bbox IoU = %.4f" % res["mean"])
        return

    res = evaluate_folder(args.pred_dir, args.gt_dir)
    for case_id, row in res["cases"].items():
        parts = []
        for c, cname in FOREGROUND.items():
            vals = row[c]
            parts.append(f"{cname[:3].upper()}({', '.join('%.3f' % v for v in vals)})")
        print(f"{case_id}:  " + "  ".join(parts))

    print(f"\nMean per class ({', '.join(METRIC_NAMES)}):")
    for c, cname in FOREGROUND.items():
        print(f"  {cname:15s}  " + "  ".join("%.4f" % v for v in res["means"][c]))


if __name__ == "__main__":
    main()
