"""
Generate synthetic 3D CT volumes + pancreas masks directly in nnU-Net v2 raw format.

Output (nnU-Net v2 convention):
    nnUNet_raw/
    └── Dataset150_PancreasCT/
        ├── imagesTr/   # <case>_0000.nii.gz
        ├── labelsTr/   # <case>.nii.gz
        ├── imagesTs/   # <case>_0000.nii.gz
        ├── labelsTs/   # <case>.nii.gz   (test GT, for evaluate.py)
        └── dataset.json

``--add_duct`` merges a thin winding duct into the pancreas label, stressing
topological continuity (relevant for Exp 3 target spacing and Exp 5 clDice).
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
)

from nnunet_utils.config import DATASET_NAME, RAW_DIR


def create_pancreas_shape(shape, center, axis_lengths, angle_deg=0, deform_scale=5.0):
    """Elongated ellipsoid (pancreas-like), rotated in the YZ plane, lightly warped."""
    D, H, W = shape
    cz, cy, cx = center
    az, ay, ax = axis_lengths

    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    Z_rot = (Z - cz) * cos_a - (Y - cy) * sin_a + cz
    Y_rot = (Z - cz) * sin_a + (Y - cy) * cos_a + cy

    dist = ((Z_rot - cz) / az) ** 2 + ((Y_rot - cy) / ay) ** 2 + ((X - cx) / ax) ** 2
    pancreas = (dist <= 1.0).astype(np.float32)

    if deform_scale > 0:
        d_field = np.random.randn(3, D, H, W).astype(np.float32) * 0.5
        for i in range(3):
            d_field[i] = gaussian_filter(d_field[i], sigma=8.0)
        coords = np.stack([Z, Y, X], axis=0) + d_field * deform_scale * 0.1
        pancreas = map_coordinates(pancreas, coords.reshape(3, -1), order=1, mode="constant")
        pancreas = (pancreas.reshape(D, H, W) > 0.5).astype(np.float32)

    return pancreas


def create_spine(shape):
    """Cylindrical spine along the Z axis."""
    D, H, W = shape
    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    cy, cx = H * 0.45, W * 0.5  # posterior
    radius = min(H, W) * 0.08
    return (((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2).astype(np.float32)


def create_vessels(shape, pancreas_mask):
    """Small vessel-like structures near the pancreas surface."""
    vessels = np.zeros(shape, dtype=np.float32)
    if not pancreas_mask.any():
        return vessels
    dist = distance_transform_edt(1 - pancreas_mask)
    surface = (dist > 0) & (dist <= 3) & (pancreas_mask == 0)
    for _ in range(2):
        surface = binary_dilation(surface, iterations=1)
    vessels[surface] = 1.0
    return vessels


def create_thin_duct(shape, pancreas_mask, thickness=1.5, jitter=2.0, n_points=20):
    """Thin winding tube through the pancreas (mimics the pancreatic duct)."""
    D, H, W = shape
    duct = np.zeros(shape, dtype=np.float32)
    if not pancreas_mask.any():
        return duct

    coords = np.argwhere(pancreas_mask > 0)
    x_min, x_max = coords[:, 2].min(), coords[:, 2].max()
    y_mean, z_mean = coords[:, 1].mean(), coords[:, 0].mean()

    xs = np.linspace(x_min, x_max, n_points)
    ys, zs = [y_mean], [z_mean]
    for _ in range(n_points - 1):
        ys.append(ys[-1] + np.random.uniform(-jitter, jitter))
        zs.append(zs[-1] + np.random.uniform(-jitter, jitter))

    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    for zc, yc, xc in zip(zs, ys, xs):
        dist = (X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2
        duct[dist <= thickness ** 2] = 1.0
    return duct


def generate_synthetic_ct(shape=(128, 192, 192), add_duct=False, duct_thickness=1.5):
    """Return (ct_volume, pancreas_mask) for one synthetic case."""
    D, H, W = shape
    ct = np.full(shape, 40.0, dtype=np.float32)
    ct += np.random.randn(*shape) * 5.0

    # background organs
    liver = create_pancreas_shape(shape, (D * 0.55, H * 0.35, W * 0.35),
                                  (D * 0.35, H * 0.28, W * 0.18), angle_deg=-10, deform_scale=3)
    ct[liver > 0.5] = np.random.normal(60, 10, size=int(liver.sum()))

    kidney_l = create_pancreas_shape(shape, (D * 0.4, H * 0.7, W * 0.25),
                                     (D * 0.12, H * 0.08, W * 0.06), angle_deg=20, deform_scale=1)
    ct[kidney_l > 0.5] = np.random.normal(70, 8, size=int(kidney_l.sum()))

    kidney_r = create_pancreas_shape(shape, (D * 0.6, H * 0.7, W * 0.75),
                                     (D * 0.12, H * 0.08, W * 0.06), angle_deg=-20, deform_scale=1)
    ct[kidney_r > 0.5] = np.random.normal(70, 8, size=int(kidney_r.sum()))

    spleen = create_pancreas_shape(shape, (D * 0.5, H * 0.3, W * 0.2),
                                   (D * 0.1, H * 0.1, W * 0.06), angle_deg=0, deform_scale=1)
    ct[spleen > 0.5] = np.random.normal(55, 10, size=int(spleen.sum()))

    spine = create_spine(shape)
    ct[spine > 0.5] = np.random.normal(300, 50, size=int(spine.sum()))

    # pancreas (target)
    pancreas = create_pancreas_shape(shape, (D * 0.5, H * 0.45, W * 0.55),
                                     (D * 0.08, H * 0.06, W * 0.14), angle_deg=-30, deform_scale=5)
    if add_duct:
        duct = create_thin_duct(shape, pancreas > 0.5, thickness=duct_thickness)
        near = binary_dilation(pancreas > 0.5, iterations=4)
        pancreas = np.maximum(pancreas > 0.5, (duct > 0.5) & near).astype(np.float32)

    ct[pancreas > 0.5] = np.random.normal(42, 6, size=int(pancreas.sum()))

    vessels = create_vessels(shape, pancreas > 0.5)
    ct[vessels > 0.5] = np.random.normal(35, 5, size=int(vessels.sum()))

    ct = gaussian_filter(ct, sigma=0.8)
    mask = (pancreas > 0.5).astype(np.uint8)
    return ct.astype(np.float32), mask


def _ct_affine(spacing=(1.5, 1.0, 1.0)):
    """Affine for a volume stored as (Z, Y, X) with typical CT spacing."""
    affine = np.eye(4)
    affine[0, 0] = spacing[0]  # z (slice thickness)
    affine[1, 1] = spacing[1]  # y
    affine[2, 2] = spacing[2]  # x
    return affine


def generate_dataset(output_dir=None, num_train=20, num_test=4, shape=(128, 192, 192),
                     seed=42, add_duct=False, duct_thickness=1.5):
    """Write a full synthetic dataset in nnU-Net v2 raw format."""
    np.random.seed(seed)

    dataset_dir = Path(output_dir) if output_dir else RAW_DIR / DATASET_NAME
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_train} train + {num_test} test cases -> {dataset_dir}")

    def _save_case(case_id, split, with_label=True):
        ct, mask = generate_synthetic_ct(shape=shape, add_duct=add_duct, duct_thickness=duct_thickness)
        affine = _ct_affine()
        nib.save(nib.Nifti1Image(ct, affine), dataset_dir / f"images{split[0:2]}" / f"{case_id}_0000.nii.gz")
        if with_label:
            nib.save(nib.Nifti1Image(mask, affine), dataset_dir / f"labels{split[0:2]}" / f"{case_id}.nii.gz")
        return mask

    for i in range(num_train):
        case_id = f"pancreas_{i:03d}"
        mask = _save_case(case_id, "Tr")
        print(f"  train {i + 1}/{num_train}: {case_id}  mask voxels={mask.sum()}")

    for i in range(num_test):
        case_id = f"pancreas_test_{i:03d}"
        mask = _save_case(case_id, "Ts")
        print(f"  test  {i + 1}/{num_test}: {case_id}  mask voxels={mask.sum()}")

    dataset_json = {
        "name": DATASET_NAME,
        "description": "Synthetic 3D CT volumes with pancreas segmentation masks",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1},
        "numTraining": num_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset written to {dataset_dir.resolve()}")
    print("Next:  python run_experiment.py --exp 1")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic pancreas CT dataset (nnU-Net raw)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_duct", action="store_true",
                        help="Merge a thin winding duct into the pancreas label")
    parser.add_argument("--duct_thickness", type=float, default=1.5)
    args = parser.parse_args()

    generate_dataset(output_dir=args.output_dir, num_train=args.num_train,
                     num_test=args.num_test, seed=args.seed,
                     add_duct=args.add_duct, duct_thickness=args.duct_thickness)
