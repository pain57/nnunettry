"""Exp 5 — nnUNetTrainer with Dice + CE + clDice (topology-preserving).

Adds a soft-skeletonized clDice term on the full-resolution prediction. It wraps
the *official* ``_build_loss`` (which already handles deep supervision), so the
Dice+CE part is untouched; only the clDice term is added on top.

Deep supervision note: when DS is enabled nnU-Net passes both the network output
and the target as *lists* (one element per supervision level). clDice is computed
only on the full-resolution head / target (index 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def soft_erode3d(x):
    """Soft erosion along each spatial axis of a (B, C, D, H, W) volume.

    max-pool3d only pools the last three (spatial) dims, hence the 3-tuple
    kernels below — the channel dim is left untouched.
    """
    p1 = -F.max_pool3d(-x, (3, 1, 1), (1, 1, 1), (1, 1, 1))
    p2 = -F.max_pool3d(-x, (1, 3, 1), (1, 1, 1), (1, 1, 1))
    p3 = -F.max_pool3d(-x, (1, 1, 3), (1, 1, 1), (1, 1, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate3d(x):
    p1 = F.max_pool3d(x, (3, 1, 1), (1, 1, 1), (1, 1, 1))
    p2 = F.max_pool3d(x, (1, 3, 1), (1, 1, 1), (1, 1, 1))
    p3 = F.max_pool3d(x, (1, 1, 3), (1, 1, 1), (1, 1, 1))
    return torch.max(torch.max(p1, p2), p3)


def soft_open3d(x):
    return soft_dilate3d(soft_erode3d(x))


def soft_skeleton3d(x):
    # s_soft = ReLU(x - soft_open(x)), as in the clDice paper
    return F.relu(x - soft_open3d(x))


class SoftclDiceLoss(nn.Module):
    """Differentiable clDice over binary masks / probability maps."""

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
    """Wraps the official DS loss and adds clDice on the full-resolution head."""

    def __init__(self, base_loss, cl_dice_loss, weight_cldice=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.cl_dice_loss = cl_dice_loss
        self.weight_cldice = weight_cldice

    def forward(self, net_output, target):
        base = self.base_loss(net_output, target)

        # DS: net_output and target are lists; use the full-resolution elements.
        pred = net_output[0] if isinstance(net_output, (list, tuple)) else net_output
        tgt = target[0] if isinstance(target, (list, tuple)) else target

        pred_prob = torch.softmax(pred, dim=1)  # (B, C, D, H, W)
        cl_dice = 0.0
        for c in range(1, pred_prob.shape[1]):  # skip background, one term per class
            pred_fg = pred_prob[:, c:c + 1]
            target_fg = (tgt[:, :1] == c).float()
            cl_dice = cl_dice + self.cl_dice_loss(target_fg, pred_fg)
        return base + self.weight_cldice * cl_dice


class nnUNetTrainer_DiceCEclDice(nnUNetTrainer):
    weight_cldice = 1.0

    def _build_loss(self):
        base_loss = super()._build_loss()  # official Dice+CE, DS-wrapped if enabled
        return AddClDiceLoss(base_loss, SoftclDiceLoss(), weight_cldice=self.weight_cldice)
