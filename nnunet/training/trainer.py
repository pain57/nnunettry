"""Training loop, patch-based data loading and validation."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Dict
from tqdm import tqdm
import numpy as np

from ..training.losses import build_loss, DeepSupervisionLoss
from ..training.lr_scheduler import PolyLRScheduler
from ..dataset.patch_sampler import PatchDataset3D
from ..dataset.augmentation import get_training_augmentation
from ..utils.metrics import dice_score


class Trainer:
    """nnU-Net trainer for 3D pancreas segmentation.

    Handles the training loop, patch-based data loading, deep supervision loss,
    and periodic validation using sliding-window inference.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        train_volumes: List[np.ndarray],
        train_masks: List[np.ndarray],
        val_volumes: Optional[List[np.ndarray]] = None,
        val_masks: Optional[List[np.ndarray]] = None,
        output_dir: str = "./runs",
        device: str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.train_volumes = train_volumes
        self.train_masks = train_masks
        self.val_volumes = val_volumes
        self.val_masks = val_masks

        self.train_aug = get_training_augmentation(p=0.5)

        # ---- Loss ----
        # Only backbones that emit auxiliary outputs (plain / resenc) get the
        # deep-supervision wrapper; transformer / ConvNeXt backbones return a
        # single logit tensor and use the base loss directly.
        base_loss = build_loss(config)
        supports_ds = bool(getattr(model, "deep_supervision", False))
        if config.deep_supervision and supports_ds:
            self.criterion = DeepSupervisionLoss(
                base_loss=base_loss, ds_weights=config.deep_supervision_weights,
            )
        else:
            self.criterion = base_loss

        # ---- Optimizer & scheduler ----
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.initial_lr,
            momentum=0.99,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
        self.scheduler = PolyLRScheduler(
            self.optimizer,
            max_epochs=config.num_epochs,
            exponent=config.poly_lr_exponent,
            initial_lr=config.initial_lr,
        )

        self.best_val_dice = 0.0
        self.current_epoch = 0

    def create_train_dataloader(self) -> DataLoader:
        dataset = PatchDataset3D(
            volumes=self.train_volumes,
            masks=self.train_masks,
            patch_size=self.config.patch_size,
            samples_per_volume=self.config.samples_per_volume,
            force_fg_ratio=self.config.force_fg_ratio,
            fg_min_ratio=self.config.fg_min_ratio,
            use_roi=self.config.use_roi,
            roi_margin=self.config.roi_margin,
            augment_fn=self.train_aug,
        )
        return DataLoader(dataset, batch_size=self.config.batch_size, num_workers=0)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        dataloader = self.create_train_dataloader()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_img, batch_seg in pbar:
            batch_img = batch_img.to(self.device)
            batch_seg = batch_seg.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(batch_img)

            if isinstance(output, tuple):
                main_logits, ds_list = output
                loss = self.criterion(main_logits, ds_list, batch_seg)
            else:
                loss = self.criterion(output, batch_seg)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on full volumes with sliding-window inference."""
        if not self.val_volumes or not self.val_masks:
            return {}

        from ..inference.predictor import SlidingWindowPredictor
        predictor = SlidingWindowPredictor(
            model=self.model,
            patch_size=self.config.patch_size,
            num_classes=self.config.num_classes,
            overlap=self.config.sliding_window_overlap,
            batch_size=self.config.sliding_window_batch_size,
            device=str(self.device),
            gaussian_sigma=self.config.gaussian_weight_sigma,
        )

        dices = []
        for vol, mask in zip(self.val_volumes, self.val_masks):
            pred = predictor.predict(vol)
            dices.append(dice_score(pred, mask))

        return {"val_dice": float(np.mean(dices)) if dices else 0.0}

    def save_checkpoint(self, filename: str):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": self.config,
        }
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop. Returns the metric history."""
        history = {"train_loss": [], "val_dice": []}

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            lr = self.scheduler.step(epoch)
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} | LR: {lr:.6f} ---")

            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            print(f"Train Loss: {train_metrics['loss']:.4f}")

            if self.val_volumes and (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                val_dice = val_metrics.get("val_dice", 0.0)
                history["val_dice"].append(val_dice)
                print(f"Val Dice: {val_dice:.4f}")

                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.save_checkpoint("checkpoint_best.pth")
                    print("  -> New best! Saved checkpoint.")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1:04d}.pth")

        self.save_checkpoint("checkpoint_final.pth")
        print(f"\nTraining complete. Best val Dice: {self.best_val_dice:.4f}")
        return history
