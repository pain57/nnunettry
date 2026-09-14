"""Shared helpers for the experiment scripts."""

import numpy as np

from nnunet.config import NNUnetConfig
from nnunet.network import build_network
from nnunet.training.trainer import Trainer
from nnunet.dataset.data_loading import load_dataset


def run_training(
    config: NNUnetConfig,
    data_dir: str,
    output_dir: str,
    device: str = "cuda",
    val_split: float = 0.0,
    pancreas_only: bool = False,
    modality: str = "ct",
    resume: str = None,
) -> Trainer:
    """Load data, build the network and train. Returns the fitted trainer."""
    print("=" * 64)
    print(f"Experiment run -> {output_dir}")
    print("=" * 64)
    print(f"  Backbone: {config.backbone} | Loss: {config.loss}")
    print(f"  Target spacing: {config.target_spacing} | ROI: {config.use_roi}")

    train_images, train_masks, _ = load_dataset(
        data_dir, config, mode="train",
        pancreas_only=pancreas_only, modality=modality,
    )

    if val_split > 0:
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(train_images))
        val_size = max(1, int(len(train_images) * val_split))
        val_idx, train_idx = indices[:val_size], indices[val_size:]
        val_images = [train_images[i] for i in val_idx]
        val_masks = [train_masks[i] for i in val_idx]
        train_images = [train_images[i] for i in train_idx]
        train_masks = [train_masks[i] for i in train_idx]
    else:
        val_images, val_masks, _ = load_dataset(
            data_dir, config, mode="validation",
            pancreas_only=pancreas_only, modality=modality,
        )

    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

    model = build_network(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    trainer = Trainer(
        model=model,
        config=config,
        train_volumes=train_images,
        train_masks=train_masks,
        val_volumes=val_images,
        val_masks=val_masks,
        output_dir=output_dir,
        device=device,
    )
    if resume:
        trainer.load_checkpoint(resume)

    trainer.train()
    print(f"Best val Dice ({output_dir}): {trainer.best_val_dice:.4f}")
    return trainer
