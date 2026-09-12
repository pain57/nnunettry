#!/usr/bin/env python3
"""
Inference script for nnU-Net 3D pancreas segmentation.

Usage:
    # Single case:
    python predict.py --input data/imagesTs/pancreas_test_000_0000.nii.gz \\
                      --output preds/ \\
                      --checkpoint runs/exp1/checkpoint_best.pth

    # Directory of cases:
    python predict.py --input data/imagesTs/ \\
                      --output preds/ \\
                      --checkpoint runs/exp1/checkpoint_best.pth
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

from nnunet.config import NNUnetConfig
from nnunet.network.unet3d import build_3d_unet
from nnunet.inference.predictor import (
    SlidingWindowPredictor,
    connected_component_postprocessing,
)
from nnunet.utils.nifti_io import load_nifti, save_segmentation_nifti
from nnunet.dataset.preprocessing import ct_intensity_normalization


def predict_single(
    input_path: str,
    output_dir: str,
    checkpoint_path: str,
    config: NNUnetConfig,
    device: str = "cuda",
    postprocess: bool = True,
):
    """Run inference on a single NIfTI file."""
    print(f"\nProcessing: {input_path}")

    # Load
    image, affine = load_nifti(input_path)
    original_shape = image.shape
    print(f"  Original shape: {original_shape}")

    # Preprocess
    image = ct_intensity_normalization(image)

    # Build model
    model = build_3d_unet(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

    # Predict
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

    # Postprocessing
    if postprocess:
        print("  Applying connected-component post-processing...")
        segmentation = connected_component_postprocessing(
            segmentation, min_volume_mm3=50.0
        )

    # Save
    case_name = Path(input_path).stem.replace("_0000", "")
    output_path = Path(output_dir) / f"{case_name}_seg.nii.gz"
    save_segmentation_nifti(segmentation, affine, str(output_path))
    print(f"  Saved: {output_path}")

    # Stats
    fg_voxels = (segmentation > 0).sum()
    fg_volume_ml = fg_voxels * 1.5 * 1.0 * 1.0 / 1000  # approximate
    print(f"  Foreground voxels: {fg_voxels} (~{fg_volume_ml:.1f} mL)")

    return segmentation


def main():
    parser = argparse.ArgumentParser(description="nnU-Net 3D pancreas segmentation inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Input NIfTI file or directory")
    parser.add_argument("--output", type=str, default="predictions",
                        help="Output directory for segmentations")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Disable connected-component post-processing")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap fraction")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("nnU-Net 3D Pancreas Segmentation — Inference")
    print("=" * 60)

    # Config
    config = NNUnetConfig()
    config.sliding_window_overlap = args.overlap
    config.sliding_window_batch_size = args.batch_size

    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        # Process all .nii.gz files
        nifti_files = sorted(input_path.glob("*.nii.gz"))
        if not nifti_files:
            print(f"No .nii.gz files found in {input_path}")
            return
        print(f"Found {len(nifti_files)} NIfTI files.")
        for nii_file in nifti_files:
            predict_single(
                str(nii_file), args.output, args.checkpoint,
                config, args.device, not args.no_postprocess,
            )
    else:
        predict_single(
            str(input_path), args.output, args.checkpoint,
            config, args.device, not args.no_postprocess,
        )

    print(f"\nAll predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
