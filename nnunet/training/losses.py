"""Loss functions: Dice, CE, clDice, deep supervision and a factory.

Experiment 5 compares ``dice_ce`` against ``dice_ce_cldice``. clDice penalizes
topological breaks (which matter for thin / tubular structures) by computing the
Dice between the soft skeletons of prediction and target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SoftDiceLoss(nn.Module):
    """Soft Dice loss (1 - Dice), averaged over foreground classes."""

    def __init__(self, apply_softmax: bool = True, skip_background: bool = True,
                 smooth: float = 1e-5):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.skip_background = skip_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_softmax:
            pred = F.softmax(logits, dim=1)
        else:
            pred = logits

        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes) \
            .permute(0, 4, 1, 2, 3).float()

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
    """Composite Dice + Cross-Entropy loss (nnU-Net default)."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        skip_background: bool = True,
        smooth: float = 1e-5,
        ce_class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(apply_softmax=True, skip_background=skip_background, smooth=smooth)
        ce_kwargs = {}
        if ce_class_weights is not None:
            ce_kwargs["weight"] = torch.tensor(ce_class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice(logits, target)
        loss_ce = self.ce(logits, target.long())
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce


# ---------------------------------------------------------------------------
# clDice (soft skeleton)
# ---------------------------------------------------------------------------

def soft_erode3d(x: torch.Tensor) -> torch.Tensor:
    """Morphological erosion via min-pooling along each axis."""
    p1 = -F.max_pool3d(-x, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate3d(x: torch.Tensor) -> torch.Tensor:
    """Morphological dilation via max-pooling along each axis."""
    p1 = F.max_pool3d(x, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = F.max_pool3d(x, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = F.max_pool3d(x, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.max(torch.max(p1, p2), p3)


def soft_open3d(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate3d(soft_erode3d(x))


def soft_skeleton3d(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable soft skeleton of a (B, 1, D, H, W) probability map."""
    skel = torch.zeros_like(x)
    for _ in range(iters):
        eroded = soft_erode3d(x)
        contour = F.relu(x - soft_dilate3d(eroded))
        skel = torch.max(skel, contour)
        x = soft_open3d(x)
    return skel


class SoftClDiceLoss(nn.Module):
    """Soft clDice loss for the foreground class (tubular-structure continuity).

    Args:
        iters: Number of skeleton-ization iterations.
        smooth: Numerical stability term.
    """

    def __init__(self, iters: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(logits, dim=1)
        # Foreground probability maps (binary pancreas segmentation).
        pred_fg = pred[:, 1:2]
        target_fg = F.one_hot(target.long(), num_classes=pred.shape[1]) \
            .permute(0, 4, 1, 2, 3).float()[:, 1:2]

        skel_pred = soft_skeleton3d(pred_fg, self.iters)
        skel_true = soft_skeleton3d(target_fg, self.iters)

        # topology precision / sensitivity
        tprec = (pred_fg * skel_true).sum() / (pred_fg.sum() + self.smooth)
        tsens = (target_fg * skel_pred).sum() / (skel_pred.sum() + self.smooth)

        cldice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cldice


class DC_CE_ClDice_Loss(nn.Module):
    """Dice + CE + clDice composite (Experiment 5)."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        cldice_weight: float = 1.0,
        cldice_iters: int = 3,
        ce_class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base = DC_and_CE_Loss(
            dice_weight=dice_weight, ce_weight=ce_weight,
            ce_class_weights=ce_class_weights,
        )
        self.cldice = SoftClDiceLoss(iters=cldice_iters)
        self.cldice_weight = cldice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base(logits, target) + self.cldice_weight * self.cldice(logits, target)


class DeepSupervisionLoss(nn.Module):
    """Apply a base loss to the main output and each auxiliary output."""

    def __init__(self, base_loss: nn.Module, ds_weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights if ds_weights is not None else [1.0, 0.5, 0.25]

    def forward(self, main_logits, ds_logits_list, target) -> torch.Tensor:
        total_loss = self.base_loss(main_logits, target)
        for i, ds_logits in enumerate(ds_logits_list):
            target_size = ds_logits.shape[2:]
            target_ds = F.interpolate(
                target.unsqueeze(1).float(), size=target_size, mode="nearest"
            ).squeeze(1).long()
            weight = self.ds_weights[min(i, len(self.ds_weights) - 1)]
            total_loss = total_loss + weight * self.base_loss(ds_logits, target_ds)
        return total_loss


def build_loss(config) -> nn.Module:
    """Build the base training criterion from a NNUnetConfig instance.

    Deep-supervision wrapping is applied separately by the trainer, since only
    some backbones (plain / resenc) emit auxiliary outputs.
    """
    if config.loss == "dice_ce":
        return DC_and_CE_Loss(
            dice_weight=config.dice_weight,
            ce_weight=config.ce_weight,
            ce_class_weights=config.ce_class_weights,
        )
    if config.loss == "dice_ce_cldice":
        return DC_CE_ClDice_Loss(
            dice_weight=config.dice_weight,
            ce_weight=config.ce_weight,
            cldice_weight=config.cldice_weight,
            cldice_iters=config.cldice_iters,
            ce_class_weights=config.ce_class_weights,
        )
    raise ValueError(f"Unknown loss '{config.loss}'. Choose 'dice_ce' or 'dice_ce_cldice'.")
