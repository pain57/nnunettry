#!/usr/bin/env python3
"""
Training script for nnU-Net 3D multi-organ (AMOS22) / pancreas segmentation.

Usage:
    # Synthetic data:
    python train.py --data_dir data --output_dir runs/exp1 --epochs 100

    # AMOS22 dataset:
    python train.py --data_dir data_amos22 --output_dir runs/amos22 \
        --epochs 1000 --num_classes 16 --batch_size 2 --device cuda

    # AMOS22 MRI-only (only use MRI cases, id >= 500):
    python train.py --data_dir data_amos22 --output_dir runs/amos22_mri \
        --epochs 1000 --num_classes 16 --modality mri --batch_size 2 --device cuda

    # AMOS22 all modalities:
    python train.py --data_dir data_amos22 --output_dir runs/amos22_all \
        --epochs 1000 --num_classes 16 --modality all --batch_size 2 --device cuda
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from nnunet.config import NNUnetConfig
from nnunet.network.unet3d import build_3d_unet
from nnunet.training.trainer import Trainer
from nnunet.utils.nifti_io import load_nifti
from nnunet.dataset.preprocessing import (
    normalize_volume,
    resample_volume,
    foreground_crop,
)


def preprocess_case(image_path: str, mask_path: str, config: NNUnetConfig,
                    pancreas_only: bool = False, modality: str = "ct"):
    """Load and preprocess a single case."""
    image, affine = load_nifti(image_path)
    mask, _ = load_nifti(mask_path)

    # Modality-specific normalization
    image = normalize_volume(image, modality=modality)

    # If pancreas-only: filter mask to keep only label 10 (pancreas)
    if pancreas_only:
        mask = (mask == 10).astype(np.int64)

    # Crop to foreground (with more margin for abdomen)
    image, mask, _ = foreground_crop(image, mask, margin=30)

    return image.astype(np.float32), mask.astype(np.int64)


def load_dataset(data_dir: str, config: NNUnetConfig, mode: str = "train",
                 pancreas_only: bool = False, modality: str = "ct"):
    """
    Load and preprocess all cases from a dataset.json file.

    Args:
        data_dir: Path to dataset root.
        config: NNUnetConfig instance.
        mode: "train", "validation", or "test".
        pancreas_only: If True, filter labels to keep only pancreas (label 10).
        modality: "ct" (id < 500), "mri" (id >= 500), or "all" (no filter).
    """
    dataset_json_path = Path(data_dir) / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(
            f"dataset.json not found in {data_dir}. "
            f"Run generate_synthetic_data.py first."
        )

    with open(dataset_json_path) as f:
        dataset_info = json.load(f)

    base_dir = Path(data_dir)

    images = []
    masks = []

    if mode in ("train", "validation"):
        key = "training" if mode == "train" else "validation"
        cases = dataset_info.get(key, [])

        for case in cases:
            img_path = base_dir / case["image"]
            lbl_path = base_dir / case["label"]

            # AMOS22: CT cases have id < 500, MRI cases have id >= 500
            case_id = int(case["image"].split("/")[-1].split("_")[-1].replace(".nii.gz", ""))
            is_mri = case_id >= 500

            if modality == "ct" and is_mri:
                print(f"  Skipped (MRI): {img_path.name}")
                continue
            if modality == "mri" and not is_mri:
                print(f"  Skipped (CT): {img_path.name}")
                continue

            if img_path.exists() and lbl_path.exists():
                img, msk = preprocess_case(str(img_path), str(lbl_path), config,
                                           pancreas_only=pancreas_only, modality=modality)
                images.append(img)
                masks.append(msk)
                tag = "MRI" if is_mri else "CT"
                print(f"  Loaded [{tag}]: {img_path.name} | shape: {img.shape} | mask voxels: {msk.sum()}")

    elif mode == "test":
        cases = dataset_info.get("test", [])
        for case in cases:
            if isinstance(case, dict):
                img_path = base_dir / case["image"]
            else:
                img_path = base_dir / case
            if img_path.exists():
                img, _ = load_nifti(str(img_path))
                img = normalize_volume(img, modality=modality)
                images.append(img.astype(np.float32))
                print(f"  Loaded test: {img_path.name} | shape: {img.shape}")

    return images, masks


def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net for 3D multi-organ segmentation")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory with dataset.json")
    parser.add_argument("--output_dir", type=str, default="runs/exp1",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--num_classes", type=int, default=16,
                        help="Number of output classes (including background)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Fraction of training data for validation "
                             "(set 0 to use dataset.json 'validation' key)")
    parser.add_argument("--pancreas_only", action="store_true",
                        help="Filter mask to keep only pancreas (label 10)")
    parser.add_argument("--modality", type=str, default="ct", choices=["ct", "mri", "all"],
                        help="Modality filter: ct (id<500), mri (id>=500), all (no filter)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # ---- Config ----
    config = NNUnetConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.initial_lr = args.lr
    config.num_classes = args.num_classes

    print("=" * 60)
    print("nnU-Net 3D Multi-Organ Segmentation — Training")
    print("=" * 60)
    print(f"Config:\n  Epochs: {config.num_epochs}\n  Batch: {config.batch_size}")
    print(f"  LR: {config.initial_lr}\n  Patch: {config.patch_size}")
    print(f"  Classes: {config.num_classes}\n  Device: {args.device}")
    print(f"  Modality: {args.modality}")
    if args.pancreas_only:
        print("  Mode: Pancreas-only (label 10)")

    # ---- Load Data ----
    print(f"\nLoading training data from {args.data_dir}...")
    train_images, train_masks = load_dataset(args.data_dir, config, mode="train",
                                              pancreas_only=args.pancreas_only,
                                              modality=args.modality)
    print(f"Loaded {len(train_images)} training cases.")

    # Load validation data
    if args.val_split > 0:
        # Random split from training
        np.random.seed(42)
        indices = np.random.permutation(len(train_images))
        val_size = max(1, int(len(train_images) * args.val_split))
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        val_images = [train_images[i] for i in val_idx]
        val_masks = [train_masks[i] for i in val_idx]
        train_images = [train_images[i] for i in train_idx]
        train_masks = [train_masks[i] for i in train_idx]
    else:
        # Use dataset.json 'validation' split (e.g. AMOS imagesVa/labelsVa)
        print(f"\nLoading validation data from {args.data_dir}...")
        val_images, val_masks = load_dataset(args.data_dir, config, mode="validation",
                                              pancreas_only=args.pancreas_only,
                                              modality=args.modality)
        print(f"Loaded {len(val_images)} validation cases.")

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

    # ---- Build Model ----
    print("\nBuilding 3D U-Net...")
    model = build_3d_unet(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        config=config,
        train_volumes=train_images,
        train_masks=train_masks,
        val_volumes=val_images,
        val_masks=val_masks,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Train ----
    print(f"\nStarting training for {config.num_epochs} epochs...")
    history = trainer.train()

    print(f"\nTraining complete! Best val Dice: {trainer.best_val_dice:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
