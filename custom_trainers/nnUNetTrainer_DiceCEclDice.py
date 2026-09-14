"""Exp 5 — Dice + CE + clDice trainer.

Adds a differentiable clDice term on top of the official Dice+CE loss to reward
topological continuity of thin / tubular structures (the pancreatic duct).

Everything about the Dice+CE base loss, its deep-supervision weighting and the
training loop comes from the official nnU-Net v2 package via
``super()._build_loss()``. The only novel piece is the clDice soft-skeleton
(min/max-pool morphological approximations of erosion/dilation, Shit et al.,
CVPR 2021), which is this experiment's contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# ---------------------------------------------------------------------------
# clDice soft-skeleton (differentiable morphological ops)
# ---------------------------------------------------------------------------
def soft_erode3d(x):
    """Soft erosion along each spatial axis (max-pool on the negative)."""
    if x.ndim == 5:  # (B, C, D, H, W)
        p1 = -F.max_pool3d(-x, (1, 3, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
        p2 = -F.max_pool3d(-x, (1, 1, 3, 1), (1, 1, 1, 1), (1, 1, 1, 1))
        p3 = -F.max_pool3d(-x, (1, 1, 1, 3), (1, 1, 1, 1), (1, 1, 1, 1))
    else:  # (B, D, H, W)
        p1 = -F.max_pool3d(-x, (3, 1, 1), (1, 1, 1), (1, 1, 1))
        p2 = -F.max_pool3d(-x, (1, 3, 1), (1, 1, 1), (1, 1, 1))
        p3 = -F.max_pool3d(-x, (1, 1, 3), (1, 1, 1), (1, 1, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate3d(x):
    if x.ndim == 5:
        p1 = F.max_pool3d(x, (1, 3, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
        p2 = F.max_pool3d(x, (1, 1, 3, 1), (1, 1, 1, 1), (1, 1, 1, 1))
        p3 = F.max_pool3d(x, (1, 1, 1, 3), (1, 1, 1, 1), (1, 1, 1, 1))
    else:
        p1 = F.max_pool3d(x, (3, 1, 1), (1, 1, 1), (1, 1, 1))
        p2 = F.max_pool3d(x, (1, 3, 1), (1, 1, 1), (1, 1, 1))
        p3 = F.max_pool3d(x, (1, 1, 3), (1, 1, 1), (1, 1, 1))
    return torch.max(torch.max(p1, p2), p3)


def soft_open3d(x):
    return soft_dilate3d(soft_erode3d(x))


def soft_skeleton3d(x):
    return F.relu(x - soft_open3d(x))


class SoftclDiceLoss(nn.Module):
    """Differentiable clDice loss on (B, 1, D, H, W) inputs in [0, 1]."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true = (y_true > 0.5).float()
        skel_pred = soft_skeleton3d(y_pred)
        skel_true = soft_skeleton3d(y_true)
        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


class AddClDiceLoss(nn.Module):
    """Wraps an existing loss and adds a clDice term on the highest-res output."""

    def __init__(self, base_loss, cl_dice_loss, weight_cldice=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.cl_dice_loss = cl_dice_loss
        self.weight_cldice = weight_cldice

    def forward(self, net_output, target):
        base = self.base_loss(net_output, target)
        # deep-supervision networks return a list; use the full-res head only
        pred = net_output[0] if isinstance(net_output, (list, tuple)) else net_output
        pred_fg = torch.softmax(pred, dim=1)[:, 1:2]     # foreground probability
        target_fg = (target[:, :1] == 1).float()          # (B, 1, D, H, W)
        return base + self.weight_cldice * self.cl_dice_loss(target_fg, pred_fg)


class nnUNetTrainer_DiceCEclDice(nnUNetTrainer):
    weight_cldice = 1.0

    def _build_loss(self):
        # Keep the official Dice+CE construction (and its deep-supervision
        # weighting) intact, then append the clDice term.
        base_loss = super()._build_loss()
        return AddClDiceLoss(base_loss, SoftclDiceLoss(), weight_cldice=self.weight_cldice)
