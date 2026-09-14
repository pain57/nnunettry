#!/usr/bin/env python3
"""Run inference through the official nnU-Net v2 predictor.

Normal (5-fold ensemble, duct segmentation on Dataset150):
    python predict.py --input nnUNet_raw/Dataset150_PancreasDuct/imagesTs \
        --output predictions --trainer nnUNetTrainer

Two-stage (Exp 4: pancreas/hepatobiliary ROI -> duct segmentation):
    python predict.py --input nnUNet_raw/Dataset150_PancreasDuct/imagesTs \
        --output predictions_c2f --coarse_to_fine --margin 20

The two-stage mode first runs the coarse ROI model (Dataset151, 3d_lowres) to
localize the pancreas/hepatobiliary region, crops each volume to that ROI, runs
the fine duct model (Dataset150, 3d_fullres) inside the crop, and pastes the duct
segmentation back into the full volume.
"""

import argparse
import subprocess
from pathlib import Path

from nnunet_utils.config import (
    DATASET_ID, DATASET_NAME, DEFAULT_PLANS, RESULTS_DIR, ROI_DATASET_NAME, setup_paths,
)

setup_paths()

ALL_FOLDS = (0, 1, 2, 3, 4)


def _run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def model_folder(trainer_name, configuration, dataset_name=DATASET_NAME,
                 plans_identifier=DEFAULT_PLANS):
    from batchgenerators.utilities.file_and_folder_operations import join
    return join(RESULTS_DIR, dataset_name, f"{trainer_name}__{plans_identifier}__{configuration}")


def predict_normal(args):
    _run(["nnUNetv2_predict",
          "-i", args.input, "-o", args.output,
          "-d", str(DATASET_ID), "-c", args.config, "-tr", args.trainer,
          "-p", args.plans, "-f"] + [str(f) for f in args.folds] + ["-chk", args.checkpoint])


def predict_coarse_to_fine(args):
    """Two-stage: coarse ROI (Dataset151) -> fine ducts (Dataset150)."""
    import nibabel as nib
    import numpy as np
    from batchgenerators.utilities.file_and_folder_operations import (
        isfile, join, maybe_mkdir_p, subfiles,
    )
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    maybe_mkdir_p(args.output)
    input_files = sorted(subfiles(args.input, suffix=".nii.gz", join=True))
    coarse_tmp = join(args.output, "_coarse")
    crop_tmp = join(args.output, "_crops")
    maybe_mkdir_p(coarse_tmp)
    maybe_mkdir_p(crop_tmp)

    # 1) coarse stage (Dataset151, 3d_lowres) — localize the pancreas/hepatobiliary ROI
    coarse = nnUNetPredictor(verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    coarse.initialize_from_trained_model_folder(
        model_folder("nnUNetTrainer", args.coarse_config, dataset_name=ROI_DATASET_NAME),
        use_folds=tuple(args.folds), checkpoint_name=args.checkpoint)
    coarse.predict_from_files(input_files, coarse_tmp, save_probabilities=False, overwrite=True)

    # 2) fine stage (Dataset150, 3d_fullres) — segment ducts inside each ROI crop
    fine = nnUNetPredictor(verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    fine.initialize_from_trained_model_folder(
        model_folder("nnUNetTrainer", args.config, dataset_name=DATASET_NAME),
        use_folds=tuple(args.folds), checkpoint_name=args.checkpoint)

    for in_file in input_files:
        case_id = Path(in_file).name.replace("_0000.nii.gz", "")
        img = nib.load(in_file)
        img_data = img.get_fdata().astype(np.float32)

        coarse_mask_path = join(coarse_tmp, f"{case_id}.nii.gz")
        cm = nib.load(coarse_mask_path).get_fdata()

        z, y, x = np.nonzero(cm > 0)
        if len(z) == 0:
            empty = nib.Nifti1Image(np.zeros_like(cm, dtype=np.uint8), img.affine)
            nib.save(empty, join(args.output, f"{case_id}.nii.gz"))
            continue

        m = args.margin
        z0, z1 = max(0, int(z.min()) - m), min(cm.shape[0], int(z.max()) + m + 1)
        y0, y1 = max(0, int(y.min()) - m), min(cm.shape[1], int(y.max()) + m + 1)
        x0, x1 = max(0, int(x.min()) - m), min(cm.shape[2], int(x.max()) + m + 1)

        crop = img_data[z0:z1, y0:y1, x0:x1]
        # keep spacing, shift the origin so the crop's voxel (0,0,0) maps back to
        # the original voxel (z0, y0, x0)
        crop_affine = img.affine.copy()
        crop_affine[:3, 3] = img.affine[:3, 3] + img.affine[:3, :3] @ np.array([z0, y0, x0])
        crop_file = join(crop_tmp, f"{case_id}_0000.nii.gz")
        nib.save(nib.Nifti1Image(crop, crop_affine), crop_file)

        fine_out = join(crop_tmp, f"{case_id}_out")
        maybe_mkdir_p(fine_out)
        fine.predict_from_files([crop_file], fine_out, save_probabilities=False, overwrite=True)
        fm_path = join(fine_out, f"{case_id}.nii.gz")
        if not isfile(fm_path):
            fm_path = join(fine_out, f"{case_id}_0000.nii.gz")
        fm = nib.load(fm_path).get_fdata()

        full = np.zeros_like(cm, dtype=fm.dtype)
        full[z0:z1, y0:y1, x0:x1] = fm
        nib.save(nib.Nifti1Image(full, img.affine), join(args.output, f"{case_id}.nii.gz"))

    print(f"\nTwo-stage (ROI -> duct) predictions saved to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="nnU-Net v2 inference")
    ap.add_argument("--input", required=True, help="folder of .nii.gz images")
    ap.add_argument("--output", required=True)
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--plans", default=DEFAULT_PLANS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(ALL_FOLDS),
                    help="folds to ensemble (default: all 5)")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--coarse_to_fine", action="store_true",
                    help="two-stage ROI -> duct inference (Exp 4)")
    ap.add_argument("--coarse_config", default="3d_lowres",
                    help="configuration used for the coarse/localization stage")
    ap.add_argument("--margin", type=int, default=20, help="ROI margin in voxels")
    args = ap.parse_args()

    if args.coarse_to_fine:
        predict_coarse_to_fine(args)
    else:
        predict_normal(args)


if __name__ == "__main__":
    main()
