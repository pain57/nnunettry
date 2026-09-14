"""Generate synthetic 3D CT volumes in nnU-Net raw format for two datasets:

* ``Dataset150_PancreasDuct`` — the main segmentation task:
  label 1 = pancreatic duct, label 2 = bile duct (thin tubes, **not** merged into
  the pancreas).
* ``Dataset151_PancreasROI`` — the Exp 4 coarse-stage target:
  label 1 = pancreas + hepatobiliary ROI (the anatomical region containing the
  ducts).

The pancreas, liver, kidneys, spleen and spine stay in the CT as *unlabeled*
background organs so the image looks realistic; only the ducts / ROI are labelled.

Spacing defaults to isotropic ``(1.0, 1.0, 1.0)`` mm so sub-mm ducts stay
resolvable. For the real dataset, spacing is decided automatically by
``nnUNetv2_plan_and_preprocess`` from the actual data — re-plan with real data
before training.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates

from nnunet_utils.config import (
    DATASET_NAME, LABELS, RAW_DIR, ROI_DATASET_NAME, ROI_LABELS,
)


def create_ellipsoid(shape, center, axis_lengths, angle_deg=0, deform_scale=5.0):
    """Elongated ellipsoid (organ-like), rotated in the ZY plane, lightly warped."""
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
    organ = (dist <= 1.0).astype(np.float32)

    if deform_scale > 0:
        d_field = np.random.randn(3, D, H, W).astype(np.float32) * 0.5
        for i in range(3):
            d_field[i] = gaussian_filter(d_field[i], sigma=8.0)
        coords = np.stack([Z, Y, X], axis=0) + d_field * deform_scale * 0.1
        organ = map_coordinates(organ, coords.reshape(3, -1), order=1, mode="constant")
        organ = (organ.reshape(D, H, W) > 0.5).astype(np.float32)

    return organ


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


def create_pancreatic_duct(shape, pancreas_mask, thickness=1.5, jitter=2.0, n_points=20):
    """Thin winding tube through the pancreas (label 1: pancreatic duct)."""
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
        duct[(X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2 <= thickness ** 2] = 1.0
    return duct


def create_bile_duct(shape, pancreas_mask, thickness=1.5, n_points=20):
    """Thin tube from the liver hilum down to the pancreatic head (label 2: bile duct)."""
    D, H, W = shape
    duct = np.zeros(shape, dtype=np.float32)
    coords = np.argwhere(pancreas_mask > 0)
    if len(coords) == 0:
        return duct

    # pancreatic head = right-most part of the pancreas (largest X)
    head = coords[coords[:, 2].argmax()]
    hz, hy, hx = float(head[0]), float(head[1]), float(head[2])
    # start near the liver hilum (upper abdomen, just right of the midline)
    sz, sy, sx = D * 0.5, H * 0.36, W * 0.42

    xs = np.linspace(sx, hx, n_points)
    ys = np.linspace(sy, hy, n_points) + np.random.uniform(-1, 1, n_points)
    zs = np.linspace(sz, hz, n_points) + np.random.uniform(-1, 1, n_points)

    Z, Y, X = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    for zc, yc, xc in zip(zs, ys, xs):
        duct[(X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2 <= thickness ** 2] = 1.0
    return duct


def generate_synthetic_ct(shape=(128, 192, 192), duct_thickness=1.5):
    """Return (ct, duct_mask, roi_mask).

    duct_mask: 0/1/2 (background / pancreatic duct / bile duct)
    roi_mask:  0/1   (pancreas + hepatobiliary ROI)
    """
    D, H, W = shape
    ct = np.full(shape, 40.0, dtype=np.float32)
    ct += np.random.randn(*shape) * 5.0

    # background organs (unlabeled in both datasets)
    liver = create_ellipsoid(shape, (D * 0.55, H * 0.35, W * 0.35),
                             (D * 0.35, H * 0.28, W * 0.18), angle_deg=-10, deform_scale=3)
    ct[liver > 0.5] = np.random.normal(60, 10, size=int(liver.sum()))

    kidney_l = create_ellipsoid(shape, (D * 0.4, H * 0.7, W * 0.25),
                                (D * 0.12, H * 0.08, W * 0.06), angle_deg=20, deform_scale=1)
    ct[kidney_l > 0.5] = np.random.normal(70, 8, size=int(kidney_l.sum()))

    kidney_r = create_ellipsoid(shape, (D * 0.6, H * 0.7, W * 0.75),
                                (D * 0.12, H * 0.08, W * 0.06), angle_deg=-20, deform_scale=1)
    ct[kidney_r > 0.5] = np.random.normal(70, 8, size=int(kidney_r.sum()))

    spleen = create_ellipsoid(shape, (D * 0.5, H * 0.3, W * 0.2),
                              (D * 0.1, H * 0.1, W * 0.06), angle_deg=0, deform_scale=1)
    ct[spleen > 0.5] = np.random.normal(55, 10, size=int(spleen.sum()))

    spine = create_spine(shape)
    ct[spine > 0.5] = np.random.normal(300, 50, size=int(spine.sum()))

    # pancreas (unlabeled background organ hosting the pancreatic duct)
    pancreas = create_ellipsoid(shape, (D * 0.5, H * 0.45, W * 0.55),
                                (D * 0.08, H * 0.06, W * 0.14), angle_deg=-30, deform_scale=5)
    ct[pancreas > 0.5] = np.random.normal(42, 6, size=int(pancreas.sum()))

    # ducts (the actual segmentation labels — kept separate, never merged)
    pd = create_pancreatic_duct(shape, pancreas > 0.5, thickness=duct_thickness)
    bd = create_bile_duct(shape, pancreas > 0.5, thickness=duct_thickness)
    for duct in (pd, bd):  # ducts read as fluid-like (low HU) on CT
        ct[duct > 0.5] = np.random.normal(25, 5, size=int(duct.sum()))

    ct = gaussian_filter(ct, sigma=0.8)

    duct_mask = np.zeros(shape, dtype=np.uint8)
    duct_mask[pd > 0.5] = 1
    duct_mask[bd > 0.5] = 2  # bile duct overwrites where the two join

    # ROI = pancreas + liver, dilated so it fully encloses the ducts
    roi = binary_dilation((pancreas > 0.5) | (liver > 0.5), iterations=3)
    roi_mask = roi.astype(np.uint8)

    return ct.astype(np.float32), duct_mask, roi_mask


def _ct_affine(spacing=(1.0, 1.0, 1.0)):
    """Affine for a volume stored as (Z, Y, X)."""
    affine = np.eye(4)
    affine[0, 0] = spacing[0]  # z
    affine[1, 1] = spacing[1]  # y
    affine[2, 2] = spacing[2]  # x
    return affine


def generate_dataset(output_dir=None, num_train=20, num_test=4, shape=(128, 192, 192),
                     seed=42, duct_thickness=1.5, spacing=(1.0, 1.0, 1.0)):
    """Write the duct dataset (Dataset150) and the ROI dataset (Dataset151)."""
    np.random.seed(seed)

    base = Path(output_dir) if output_dir else RAW_DIR
    duct_dir = base / DATASET_NAME
    roi_dir = base / ROI_DATASET_NAME
    for d in (duct_dir, roi_dir):
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (d / sub).mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_train} train + {num_test} test cases ->\n"
          f"  ducts: {duct_dir}\n  roi:   {roi_dir}")

    def _save_case(case_id, split, with_label=True):
        ct, duct_mask, roi_mask = generate_synthetic_ct(shape=shape, duct_thickness=duct_thickness)
        affine = _ct_affine(spacing)

        for d, mask in ((duct_dir, duct_mask), (roi_dir, roi_mask)):
            nib.save(nib.Nifti1Image(ct, affine), d / f"images{split[0:2]}" / f"{case_id}_0000.nii.gz")
            if with_label:
                nib.save(nib.Nifti1Image(mask, affine), d / f"labels{split[0:2]}" / f"{case_id}.nii.gz")
        return duct_mask, roi_mask

    for i in range(num_train):
        case_id = f"duct_{i:03d}"
        dm, rm = _save_case(case_id, "Tr")
        print(f"  train {i + 1}/{num_train}: {case_id}  "
              f"pd={(dm == 1).sum()} bd={(dm == 2).sum()} roi={rm.sum()}")

    for i in range(num_test):
        case_id = f"duct_test_{i:03d}"
        dm, rm = _save_case(case_id, "Ts")
        print(f"  test  {i + 1}/{num_test}: {case_id}  "
              f"pd={(dm == 1).sum()} bd={(dm == 2).sum()} roi={rm.sum()}")

    _write_dataset_json(duct_dir, DATASET_NAME, LABELS, num_train,
                        "Synthetic 3D CT with pancreatic-duct and bile-duct masks")
    _write_dataset_json(roi_dir, ROI_DATASET_NAME, ROI_LABELS, num_train,
                        "Synthetic 3D CT with pancreas + hepatobiliary ROI masks")

    print(f"\nDatasets written to {base.resolve()}")
    print("Next:  python run_experiment.py --exp 1")


def _write_dataset_json(dataset_dir, name, labels, num_train, description):
    dataset_json = {
        "name": name,
        "description": description,
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_train,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CT duct + ROI datasets (nnU-Net raw)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duct_thickness", type=float, default=1.5)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        help="voxel spacing (z y x) in mm; use the real-data spacing for training")
    args = parser.parse_args()

    generate_dataset(output_dir=args.output_dir, num_train=args.num_train,
                     num_test=args.num_test, seed=args.seed,
                     duct_thickness=args.duct_thickness, spacing=tuple(args.spacing))
