"""
Generate synthetic 3D CT volumes and pancreas segmentation masks.

Creates simplified pancreas-like shapes (elongated ellipsoids with slight
deformation) embedded in a CT-like background with surrounding organ-like
structures, spinal column, etc. Useful for testing the nnU-Net pipeline.

Output structure (nnU-Net convention):
    data/
    ├── imagesTr/    # training images  (.nii.gz)
    ├── labelsTr/    # training labels   (.nii.gz)
    ├── imagesTs/    # test images       (.nii.gz)
    └── dataset.json # metadata
"""

import numpy as np
import nibabel as nib
import json
import os
from pathlib import Path
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    distance_transform_edt,
    map_coordinates,
)


def create_pancreas_shape(shape, center, axis_lengths, angle_deg=0, deform_scale=5.0):
    """
    Create an elongated ellipsoid (pancreas-like) at `center` with
    `axis_lengths` along (Z, Y, X), rotated by `angle_deg` in the YZ plane
    (coronal rotation), with slight random elastic deformation.
    """
    D, H, W = shape
    cz, cy, cx = center
    az, ay, ax = axis_lengths

    # Coordinate grid
    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    # Rotation in YZ plane (coronal)
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    Z_rot = (Z - cz) * cos_a - (Y - cy) * sin_a + cz
    Y_rot = (Z - cz) * sin_a + (Y - cy) * cos_a + cy

    # Ellipsoid equation
    dist = (
        ((Z_rot - cz) / az) ** 2
        + ((Y_rot - cy) / ay) ** 2
        + ((X - cx) / ax) ** 2
    )

    pancreas = (dist <= 1.0).astype(np.float32)

    # Small elastic-like deformation to make it more realistic
    if deform_scale > 0:
        # Create a small random displacement field and warp
        d_field = np.random.randn(3, D, H, W).astype(np.float32) * 0.5
        for i in range(3):
            d_field[i] = gaussian_filter(d_field[i], sigma=8.0)

        coords = np.stack([Z, Y, X], axis=0) + d_field * deform_scale * 0.1
        pancreas = map_coordinates(
            pancreas, coords.reshape(3, -1), order=1, mode="constant"
        ).reshape(D, H, W)
        pancreas = (pancreas > 0.5).astype(np.float32)

    return pancreas


def create_spine(shape):
    """Create a cylindrical spine along the Z axis."""
    D, H, W = shape
    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    cy, cx = H * 0.45, W * 0.5  # posterior
    radius = min(H, W) * 0.08
    spine = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius**2
    return spine.astype(np.float32)


def create_vessels(shape, pancreas_mask):
    """Create small vessel-like structures near the pancreas."""
    D, H, W = shape
    vessels = np.zeros(shape, dtype=np.float32)

    if not pancreas_mask.any():
        return vessels

    # Find pancreas surface
    dist = distance_transform_edt(1 - pancreas_mask)
    surface_mask = (dist > 0) & (dist <= 3) & (pancreas_mask == 0)

    # Dilate slightly along Z
    vessel_seeds = np.copy(surface_mask)
    for _ in range(2):
        vessel_seeds = binary_dilation(vessel_seeds, iterations=1)

    vessels[vessel_seeds] = 1.0
    return vessels


def generate_synthetic_ct(shape=(128, 192, 192)):
    """
    Generate one synthetic CT volume with pancreas label.

    Returns:
        (ct_volume, pancreas_mask)
    """
    D, H, W = shape

    # Background with soft tissue CT values (~30-50 HU)
    ct = np.full(shape, 40.0, dtype=np.float32)
    ct += np.random.randn(*shape) * 5.0  # noise

    # ---- Organs (background structures) ----
    # Liver (right side, large)
    liver_center = (D * 0.55, H * 0.35, W * 0.35)
    liver_axes = (D * 0.35, H * 0.28, W * 0.18)
    liver = create_pancreas_shape(shape, liver_center, liver_axes, angle_deg=-10, deform_scale=3)
    ct[liver > 0.5] = np.random.normal(60, 10, size=int(liver.sum()))

    # Left kidney
    kidney_l_center = (D * 0.4, H * 0.7, W * 0.25)
    kidney_l_axes = (D * 0.12, H * 0.08, W * 0.06)
    kidney_l = create_pancreas_shape(shape, kidney_l_center, kidney_l_axes, angle_deg=20, deform_scale=1)
    ct[kidney_l > 0.5] = np.random.normal(70, 8, size=int(kidney_l.sum()))

    # Right kidney
    kidney_r_center = (D * 0.6, H * 0.7, W * 0.75)
    kidney_r_axes = (D * 0.12, H * 0.08, W * 0.06)
    kidney_r = create_pancreas_shape(shape, kidney_r_center, kidney_r_axes, angle_deg=-20, deform_scale=1)
    ct[kidney_r > 0.5] = np.random.normal(70, 8, size=int(kidney_r.sum()))

    # Spleen (left side)
    spleen_center = (D * 0.5, H * 0.3, W * 0.2)
    spleen_axes = (D * 0.1, H * 0.1, W * 0.06)
    spleen = create_pancreas_shape(shape, spleen_center, spleen_axes, angle_deg=0, deform_scale=1)
    ct[spleen > 0.5] = np.random.normal(55, 10, size=int(spleen.sum()))

    # ---- Spine ----
    spine = create_spine(shape)
    ct[spine > 0.5] = np.random.normal(300, 50, size=int(spine.sum()))

    # ---- Pancreas (target) ----
    # Pancreas located near the center, between liver and spine
    pancreas_center = (D * 0.5, H * 0.45, W * 0.55)
    pancreas_axes = (D * 0.08, H * 0.06, W * 0.14)  # elongated along X (left-right)
    pancreas = create_pancreas_shape(
        shape, pancreas_center, pancreas_axes, angle_deg=-30, deform_scale=5
    )

    # Set pancreas CT values (~35-50 HU, slightly hypodense vs liver)
    ct[pancreas > 0.5] = np.random.normal(42, 6, size=int(pancreas.sum()))

    # ---- Vessels near pancreas ----
    vessels = create_vessels(shape, pancreas)
    ct[vessels > 0.5] = np.random.normal(35, 5, size=int(vessels.sum()))

    # ---- Smooth the CT ----
    ct = gaussian_filter(ct, sigma=0.8)

    # Binarize pancreas mask
    pancreas_mask = (pancreas > 0.5).astype(np.uint8)

    return ct.astype(np.float32), pancreas_mask


def generate_dataset(
    output_dir: str = "data",
    num_train: int = 20,
    num_test: int = 4,
    shape: tuple = (128, 192, 192),
    seed: int = 42,
):
    """Generate a full synthetic pancreas CT dataset."""
    np.random.seed(seed)

    output_dir = Path(output_dir)
    images_tr_dir = output_dir / "imagesTr"
    labels_tr_dir = output_dir / "labelsTr"
    images_ts_dir = output_dir / "imagesTs"

    for d in [images_tr_dir, labels_tr_dir, images_ts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_train} training + {num_test} test synthetic CT volumes...")

    # ---- Training data ----
    for i in range(num_train):
        case_id = f"pancreas_{i:03d}"
        ct, mask = generate_synthetic_ct(shape=shape)

        # Save as NIfTI
        affine = np.eye(4)
        spacing = (1.5, 1.0, 1.0)  # typical CT spacing
        affine[0, 0] = spacing[2]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[0]

        img_nii = nib.Nifti1Image(ct, affine)
        mask_nii = nib.Nifti1Image(mask.astype(np.uint8), affine)

        nib.save(img_nii, str(images_tr_dir / f"{case_id}_0000.nii.gz"))
        nib.save(mask_nii, str(labels_tr_dir / f"{case_id}.nii.gz"))
        print(f"  Train {i + 1}/{num_train}: {case_id} | mask voxels: {mask.sum()}")

    # ---- Test data ----
    for i in range(num_test):
        case_id = f"pancreas_test_{i:03d}"
        ct, mask = generate_synthetic_ct(shape=shape)

        affine = np.eye(4)
        spacing = (1.5, 1.0, 1.0)
        affine[0, 0] = spacing[2]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[0]

        img_nii = nib.Nifti1Image(ct, affine)
        nib.save(img_nii, str(images_ts_dir / f"{case_id}_0000.nii.gz"))
        print(f"  Test {i + 1}/{num_test}: {case_id}")

    # ---- dataset.json ----
    dataset_json = {
        "name": "SyntheticPancreasCT",
        "description": "Synthetic 3D CT volumes with pancreas segmentation masks",
        "reference": "synthetic",
        "licence": "CC0",
        "release": "1.0",
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "pancreas": 1,
        },
        "numTraining": num_train,
        "numTest": num_test,
        "training": [
            {"image": f"./imagesTr/pancreas_{i:03d}_0000.nii.gz",
             "label": f"./labelsTr/pancreas_{i:03d}.nii.gz"}
            for i in range(num_train)
        ],
        "test": [
            f"./imagesTs/pancreas_test_{i:03d}_0000.nii.gz"
            for i in range(num_test)
        ],
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset saved to {output_dir.resolve()}")
    print(f"  Training: {num_train} cases")
    print(f"  Test:     {num_test} cases")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic pancreas CT dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--num_train", type=int, default=20,
                        help="Number of training cases")
    parser.add_argument("--num_test", type=int, default=4,
                        help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed,
    )
