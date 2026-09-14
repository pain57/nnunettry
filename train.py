#!/usr/bin/env python3
"""Training entry point for all six pancreas-segmentation experiments.

The experiment is fully described by the command-line flags, which map to
fields of :class:`nnunet.config.NNUnetConfig`:

    python train.py --data_dir data --output_dir runs/exp1_baseline \
        --backbone plain --epochs 1000

    # Experiment 2 (residual encoder):
    python train.py --data_dir data --output_dir runs/exp2_resenc_m --backbone resenc_m

    # Experiment 3 (target spacing):
    python train.py --data_dir data --output_dir runs/exp3_1mm \
        --target_spacing 1.0,1.0,1.0

    # Experiment 4 (ROI-focused fine stage):
    python train.py --data_dir data --output_dir runs/exp4_fine --use_roi

    # Experiment 5 (continuity loss):
    python train.py --data_dir data --output_dir runs/exp5_cldice --loss dice_ce_cldice

    # Experiment 6 (other architectures):
    python train.py --data_dir data --output_dir runs/exp6_unetr --backbone unetr
"""

import argparse
import numpy as np

from nnunet.config import NNUnetConfig
from nnunet.network import build_network, BACKBONE_CHOICES
from nnunet.training.trainer import Trainer
from nnunet.dataset.data_loading import load_dataset


def parse_spacing(s: str):
    """Parse a 'a,b,c' string into a tuple of floats."""
    try:
        return tuple(float(x) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"target spacing must be 'a,b,c' (got {s!r})"
        )


def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net for 3D pancreas segmentation")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset directory with dataset.json")
    parser.add_argument("--output_dir", type=str, default="runs/exp1", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=2, help="Classes incl. background")
    parser.add_argument("--backbone", type=str, default="plain", choices=BACKBONE_CHOICES)
    parser.add_argument("--target_spacing", type=parse_spacing, default=None,
                        help="Resample to voxel spacing, e.g. 1.0,1.0,1.0 (Exp3)")
    parser.add_argument("--loss", type=str, default="dice_ce",
                        choices=["dice_ce", "dice_ce_cldice"])
    parser.add_argument("--cldice_weight", type=float, default=1.0)
    parser.add_argument("--cldice_iters", type=int, default=3)
    parser.add_argument("--use_roi", action="store_true",
                        help="Sample training patches inside the GT pancreas ROI (Exp4)")
    parser.add_argument("--roi_margin", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Fraction of training data for validation")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--pancreas_only", action="store_true",
                        help="Filter labels to keep only pancreas (label 10)")
    parser.add_argument("--modality", type=str, default="ct", choices=["ct", "mri", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = NNUnetConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.initial_lr = args.lr
    config.num_classes = args.num_classes
    config.backbone = args.backbone
    config.target_spacing = args.target_spacing
    config.loss = args.loss
    config.cldice_weight = args.cldice_weight
    config.cldice_iters = args.cldice_iters
    config.use_roi = args.use_roi
    config.roi_margin = args.roi_margin
    config.val_every = args.val_every

    print("=" * 64)
    print("nnU-Net 3D Pancreas Segmentation — Training")
    print("=" * 64)
    print(f"  Backbone: {config.backbone} | Loss: {config.loss} | Classes: {config.num_classes}")
    print(f"  Target spacing: {config.target_spacing}")
    print(f"  ROI sampling: {config.use_roi} (margin {config.roi_margin})")
    print(f"  Epochs: {config.num_epochs} | Batch: {config.batch_size} | LR: {config.initial_lr}")
    print(f"  Device: {args.device}")

    # ---- Data ----
    print(f"\nLoading training data from {args.data_dir}...")
    train_images, train_masks, _ = load_dataset(
        args.data_dir, config, mode="train",
        pancreas_only=args.pancreas_only, modality=args.modality,
    )
    print(f"Loaded {len(train_images)} training cases.")

    if args.val_split > 0:
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(train_images))
        val_size = max(1, int(len(train_images) * args.val_split))
        val_idx, train_idx = indices[:val_size], indices[val_size:]
        val_images = [train_images[i] for i in val_idx]
        val_masks = [train_masks[i] for i in val_idx]
        train_images = [train_images[i] for i in train_idx]
        train_masks = [train_masks[i] for i in train_idx]
    else:
        print(f"\nLoading validation data from {args.data_dir}...")
        val_images, val_masks, _ = load_dataset(
            args.data_dir, config, mode="validation",
            pancreas_only=args.pancreas_only, modality=args.modality,
        )
        print(f"Loaded {len(val_images)} validation cases.")

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

    # ---- Model ----
    print(f"\nBuilding backbone '{config.backbone}'...")
    model = build_network(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ---- Train ----
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

    print(f"\nStarting training for {config.num_epochs} epochs...")
    history = trainer.train()

    print(f"\nTraining complete! Best val Dice: {trainer.best_val_dice:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
