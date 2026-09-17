#!/usr/bin/env python3
"""Convert AMOS22 **MRI** cases into an nnU-Net v2 raw dataset (pancreas only).

AMOS22 (https://amos22.grand-challenge.org/) ships CT and MRI in separate
folders. Only the MRI subset is converted here:

* AMOS22 pancreas organ label ``10`` -> binary ``1`` (everything else -> 0);
* output follows the nnU-Net v2 layout: ``imagesTr`` / ``labelsTr`` (and
  optionally ``imagesTs`` / ``labelsTs``) + ``dataset.json``;
* ``dataset.json`` modality is ``"MRI"`` (not CT).

Usage (training split only):
    python convert_amos_mri_pancreas.py \
        --images_dir /path/to/amos22_mri/imagesTr \
        --labels_dir /path/to/amos22_mri/labelsTr

Usage (training + test split):
    python convert_amos_mri_pancreas.py \
        --images_dir /path/to/amos22_mri/imagesTr \
        --labels_dir /path/to/amos22_mri/labelsTr \
        --test_images_dir /path/to/amos22_mri/imagesVa \
        --test_labels_dir /path/to/amos22_mri/labelsVa
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet_utils.config import DATASET_NAME, LABELS, RAW_DIR

#: AMOS22 organ label used for the pancreas.
PANCREAS_LABEL = 10

#: modality recorded in dataset.json (MRI, not CT).
MODALITY = "MRI"

IMAGE_READER_WRITER = "SimpleITKIO"


def _nifti_files(d):
    d = Path(d)
    return sorted([p for p in d.glob("*.nii.gz") if not p.name.endswith("_0000.nii.gz")]
                  or list(d.glob("*.nii.gz")))


def _case_id(name):
    # strip .nii.gz and an optional _0000 channel suffix
    base = name[:-7] if name.endswith(".nii.gz") else name
    return base[:-5] if base.endswith("_0000") else base


def _remap_pancreas(label_file):
    label = nib.load(label_file)
    data = np.asanyarray(label.dataobj)
    data = data.astype(np.int16)
    binary = (data == PANCREAS_LABEL).astype(np.uint8)
    if binary.sum() == 0:
        print(f"  WARNING: no pancreas (label {PANCREAS_LABEL}) found in {label_file.name}")
    return nib.Nifti1Image(binary, label.affine, label.header)


def _convert_split(img_dir, lab_dir, out_img_sub, out_lab_sub, out_dir):
    """Convert one split (train or test) of AMOS MRI into nnU-Net raw folders."""
    img_dir, lab_dir = Path(img_dir), Path(lab_dir)
    out_img = out_dir / out_img_sub
    out_lab = out_dir / out_lab_sub
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    images = _nifti_files(img_dir)
    if not images:
        raise FileNotFoundError(f"no MRI images found in {img_dir}")

    n_done = 0
    for img in images:
        cid = _case_id(img.name)
        lab = lab_dir / f"{cid}.nii.gz"
        if not lab.exists():
            print(f"  WARNING: no label for {img.name}, skipping")
            continue

        # verify shapes match before writing anything
        img_shape = nib.load(img).shape
        lab_shape = nib.load(lab).shape
        if img_shape != lab_shape:
            print(f"  WARNING: shape mismatch {cid} (img {img_shape} vs label {lab_shape}), skipping")
            continue

        shutil.copyfile(img, out_img / f"{cid}_0000.nii.gz")          # image keeps channel suffix
        nib.save(_remap_pancreas(lab), out_lab / f"{cid}.nii.gz")     # label 10 -> 1
        n_done += 1

    print(f"  {out_img_sub}: {n_done} cases")
    return n_done


def _write_dataset_json(out_dir, num_train):
    dataset_json = {
        "name": DATASET_NAME,
        "description": "AMOS22 MRI — pancreas segmentation (organ label 10 mapped to 1)",
        "channel_names": {"0": MODALITY},
        "labels": LABELS,
        "numTraining": num_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": IMAGE_READER_WRITER,
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"  dataset.json written ({MODALITY}, {num_train} training cases)")


def main():
    ap = argparse.ArgumentParser(description="Convert AMOS22 MRI -> nnU-Net raw (pancreas)")
    ap.add_argument("--images_dir", required=True, help="AMOS MRI imagesTr folder")
    ap.add_argument("--labels_dir", required=True, help="AMOS MRI labelsTr folder")
    ap.add_argument("--test_images_dir", default=None, help="optional AMOS MRI imagesVa folder")
    ap.add_argument("--test_labels_dir", default=None, help="optional AMOS MRI labelsVa folder")
    ap.add_argument("--output_dir", default=None,
                    help=f"default: {RAW_DIR / DATASET_NAME}")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else RAW_DIR / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting AMOS22 MRI -> {out_dir}")
    n_train = _convert_split(args.images_dir, args.labels_dir,
                             "imagesTr", "labelsTr", out_dir)

    n_test = 0
    if args.test_images_dir and args.test_labels_dir:
        n_test = _convert_split(args.test_images_dir, args.test_labels_dir,
                                "imagesTs", "labelsTs", out_dir)

    _write_dataset_json(out_dir, n_train)

    print(f"\nDone: {n_train} train + {n_test} test cases in {out_dir.resolve()}")
    print("Next:  nnUNetv2_plan_and_preprocess -d 150 -c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
