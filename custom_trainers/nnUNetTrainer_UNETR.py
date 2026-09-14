"""Exp 6 — UNETR (MONAI) trainer.

Single-output network (no deep-supervision heads), so the loss is the plain
official Dice+CE *without* the DeepSupervisionWrapper. Deep supervision is
disabled for this experiment in the plans file (``disable_deep_supervision``).

The input size is read directly from ``self.configuration_manager.patch_size``
(2.8.1 trainer interface) instead of being injected into the plans file.
"""

import torch.nn as nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_UNETR(nnUNetTrainer):
    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision) -> nn.Module:
        from monai.networks.nets import UNETR
        img_size = tuple(self.configuration_manager.patch_size)
        return UNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
        )

    def _build_loss(self):
        # No deep supervision: plain Dice+CE, no DeepSupervisionWrapper.
        return DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)
