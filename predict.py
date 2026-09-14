#!/usr/bin/env python3
"""Inference entry point.

Supports single-stage inference and coarse-to-fine (Experiment 4) via
``--coarse_checkpoint``:

    python predict.py --input data/imagesTs --output preds \
        --checkpoint runs/exp1_baseline/checkpoint_best.pth

    python predict.py --input data/imagesTs --output preds \
        --checkpoint runs/exp4_fine/checkpoint_best.pth \
        --coarse_checkpoint runs/exp4_coarse/checkpoint_best.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

from nnunet.config import NNUnetConfig
from nnunet.network import build_network
from nnunet.inference.predictor import (
    SlidingWindowPredictor,
    connected_component_postprocessing,
)
from nnunet.inference.coarse_to_fine import CoarseToFinePredictor
from nnunet.utils.nifti_io import load_nifti_with_spacing, save_segmentation_nifti
from nnunet.dataset.preprocessing import ct_intensity_normalization, resample_volume


def load_model(checkpoint_path: str, device: str):
    """Load a model (and its config) from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", NNUnetConfig())
    model = build_network(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, backbone {config.backbone})")
    return model, config


def predict_single(
    input_path: str,
    output_dir: str,
    model,
    config: NNUnetConfig,
    device: str = "cuda",
    postprocess: bool = True,
    coarse_model=None,
):
    """Run inference on a single NIfTI file (optionally coarse-to-fine)."""
    print(f"\nProcessing: {input_path}")

    image, affine, spacing = load_nifti_with_spacing(input_path)
    original_shape = image.shape
    print(f"  Original shape: {original_shape} | spacing: {spacing}")

    image = ct_intensity_normalization(image)

    # Match the training preprocessing: resample to target spacing if requested.
    if config.target_spacing is not None:
        image = resample_volume(image, spacing, config.target_spacing,
                                is_label=False, order=config.resample_order_image)

    # ---- Predict ----
    if coarse_model is not None:
        coarse, _ = coarse_model
        predictor = CoarseToFinePredictor(
            coarse_model=coarse,
            fine_model=model,
            coarse_patch_size=config.patch_size,
            fine_patch_size=config.patch_size,
            num_classes=config.num_classes,
            roi_margin=config.roi_margin,
            overlap=config.sliding_window_overlap,
            batch_size=config.sliding_window_batch_size,
            device=device,
            gaussian_sigma=config.gaussian_weight_sigma,
        )
    else:
        predictor = SlidingWindowPredictor(
            model=model,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            overlap=config.sliding_window_overlap,
            batch_size=config.sliding_window_batch_size,
            device=device,
            gaussian_sigma=config.gaussian_weight_sigma,
        )

    segmentation = predictor.predict(image)

    # Resample the segmentation back to the original spacing/shape.
    if config.target_spacing is not None and segmentation.shape != original_shape:
        back_zoom = tuple(o / s for o, s in zip(original_shape, segmentation.shape))
        segmentation = zoom(segmentation.astype(np.float32), back_zoom, order=0, prefilter=False)
        segmentation = np.round(segmentation).astype(np.uint8)

    if postprocess:
        print("  Applying connected-component post-processing...")
        segmentation = connected_component_postprocessing(
            segmentation, min_volume_mm3=50.0, spacing=spacing,
        )

    # ---- Save ----
    case_name = Path(input_path).stem.replace("_0000", "")
    output_path = Path(output_dir) / f"{case_name}_seg.nii.gz"
    save_segmentation_nifti(segmentation, affine, str(output_path))
    print(f"  Saved: {output_path}")

    fg_voxels = int((segmentation > 0).sum())
    print(f"  Foreground voxels: {fg_voxels}")
    return segmentation


def main():
    parser = argparse.ArgumentParser(description="nnU-Net 3D pancreas segmentation inference")
    parser.add_argument("--input", type=str, required=True, help="Input NIfTI file or directory")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coarse_checkpoint", type=str, default=None,
                        help="Coarse model for coarse-to-fine inference (Exp4)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print("=" * 64)
    print("nnU-Net 3D Pancreas Segmentation — Inference")
    print("=" * 64)

    model, config = load_model(args.checkpoint, args.device)
    config.sliding_window_overlap = args.overlap
    config.sliding_window_batch_size = args.batch_size

    coarse = None
    if args.coarse_checkpoint:
        coarse_model, coarse_config = load_model(args.coarse_checkpoint, args.device)
        coarse = (coarse_model, coarse_config)

    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        nifti_files = sorted(input_path.glob("*.nii.gz"))
        if not nifti_files:
            print(f"No .nii.gz files found in {input_path}")
            return
        print(f"Found {len(nifti_files)} NIfTI files.")
        for nii_file in nifti_files:
            predict_single(str(nii_file), args.output, model, config,
                           args.device, not args.no_postprocess, coarse)
    else:
        predict_single(str(input_path), args.output, model, config,
                       args.device, not args.no_postprocess, coarse)

    print(f"\nAll predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
