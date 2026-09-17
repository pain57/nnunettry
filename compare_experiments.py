#!/usr/bin/env python3
"""Formal comparison of experiment predictions (Dice/Recall/Precision/HD95/ASSD).

Point each experiment at its prediction folder and get a single comparison table.
HD95 / ASSD are in mm (real spacing from the ground-truth headers).

    python compare_experiments.py \
        --gt_dir nnUNet_raw/Dataset150_AMOSMRI_Pancreas/labelsTs \
        --pred_dirs \
            exp1=predictions/exp1 \
            exp2_resencM=predictions/exp2_resencM \
            exp2_resencL=predictions/exp2_resencL \
            exp6_unetr=predictions/exp6_unetr \
            exp6_swinunetr=predictions/exp6_swinunetr \
            exp6_mednext=predictions/exp6_mednext
"""

import argparse

import numpy as np

from nnunet_utils.evaluation import FOREGROUND, METRIC_NAMES, evaluate_folder


def main():
    ap = argparse.ArgumentParser(description="Compare experiment predictions")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dirs", nargs="+", required=True,
                    help="name=path pairs, e.g. exp1=predictions/exp1 exp2=predictions/exp2")
    args = ap.parse_args()

    # header: one column per (class x metric) plus an overall mean per metric
    header = ["experiment"]
    for _, cname in FOREGROUND.items():
        header += [f"{cname[:3]}.{m}" for m in METRIC_NAMES]
    header += [f"mean.{m}" for m in METRIC_NAMES]

    rows = []
    for item in args.pred_dirs:
        name, path = item.split("=", 1)
        res = evaluate_folder(path, args.gt_dir)
        means = res["means"]

        row = [name]
        for c in FOREGROUND:
            row += list(means[c])
        overall = np.nanmean(np.array([means[c] for c in FOREGROUND]), axis=0)
        row += list(overall)
        rows.append(row)

    def _fmt(v, i):
        return v if i == 0 else "%.4f" % v

    print("  ".join(f"{h:<15}" for h in header))
    print("-" * (len(header) * 17))
    for row in rows:
        print("  ".join(f"{_fmt(v, i):<15}" for i, v in enumerate(row)))


if __name__ == "__main__":
    main()
