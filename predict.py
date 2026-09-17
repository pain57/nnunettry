#!/usr/bin/env python3
"""Run inference through the official nnU-Net v2 predictor (AMOS22 MRI pancreas).

    python predict.py --input nnUNet_raw/Dataset150_AMOSMRI_Pancreas/imagesTs \
        --output predictions --trainer nnUNetTrainer

By default it ensembles all 5 folds.
"""

import argparse
import subprocess

from nnunet_utils.config import DATASET_ID, DEFAULT_PLANS, setup_paths

setup_paths()

ALL_FOLDS = (0, 1, 2, 3, 4)


def _run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="nnU-Net v2 inference (AMOS MRI pancreas)")
    ap.add_argument("--input", required=True, help="folder of .nii.gz images")
    ap.add_argument("--output", required=True)
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--plans", default=DEFAULT_PLANS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(ALL_FOLDS),
                    help="folds to ensemble (default: all 5)")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    args = ap.parse_args()

    _run(["nnUNetv2_predict", "-i", args.input, "-o", args.output,
          "-d", str(DATASET_ID), "-c", args.config, "-tr", args.trainer,
          "-p", args.plans, "-f"] + [str(f) for f in args.folds] + ["-chk", args.checkpoint])


if __name__ == "__main__":
    main()
