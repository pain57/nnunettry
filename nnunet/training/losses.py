import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.
    Loss = 1 - Dice averaged over all classes (excluding background by default).
    """

    def __init__(self, apply_softmax: bool = True, skip_background: bool = True, smooth: float = 1e-5):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.skip_background = skip_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw logits.
            target: (B, D, H, W) integer labels.
        """
        if self.apply_softmax:
            pred = F.softmax(logits, dim=1)
        else:
            pred = logits

        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        # (B, C, D, H, W)

        start_class = 1 if self.skip_background else 0
        dice_sum = 0.0
        count = 0
        for c in range(start_class, num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            intersection = (pred_c * target_c).sum(dim=(1, 2, 3))
            denom = pred_c.sum(dim=(1, 2, 3)) + target_c.sum(dim=(1, 2, 3))
            dice_c = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            dice_sum += dice_c.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)
        return 1.0 - dice_sum / count


class DC_and_CE_Loss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss (same as nnU-Net default).
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        skip_background: bool = True,
        smooth: float = 1e-5,
        ce_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(apply_softmax=True, skip_background=skip_background, smooth=smooth)
        ce_kwargs = ce_kwargs or {}
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice(logits, target)
        loss_ce = self.ce(logits, target.long())
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce


class DeepSupervisionLoss(nn.Module):
    """
    Wraps a base loss and applies it to the main output and each
    deep supervision output with exponentially decaying weights.
    Loss = base_loss(main) + Σ w_i * base_loss(ds_i)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        ds_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights if ds_weights is not None else [1.0, 0.5, 0.25]

    def forward(
        self,
        main_logits: torch.Tensor,
        ds_logits_list: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = self.base_loss(main_logits, target)

        for i, ds_logits in enumerate(ds_logits_list):
            # Downsample target to match deep supervision resolution
            target_size = ds_logits.shape[2:]  # (D, H, W)
            target_ds = F.interpolate(
                target.unsqueeze(1).float(),
                size=target_size,
                mode="nearest",
            ).squeeze(1).long()
            ds_loss = self.base_loss(ds_logits, target_ds)
            weight = self.ds_weights[min(i, len(self.ds_weights) - 1)]
            total_loss = total_loss + weight * ds_loss

        return total_loss
