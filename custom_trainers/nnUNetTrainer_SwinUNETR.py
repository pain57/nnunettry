"""Exp 6 — SwinUNETR (MONAI) trainer.

Single-output network (no deep-supervision heads); same plain Dice+CE loss and
deep-supervision-off setup as the UNETR trainer (see its docstring).
"""

import torch.nn as nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_SwinUNETR(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision) -> nn.Module:
        from monai.networks.nets import SwinUNETR
        img_size = tuple(arch_init_kwargs.get("img_size", (96, 96, 96)))
        return SwinUNETR(
            img_size=img_size,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
        )

    def _build_loss(self):
        # No deep supervision: plain Dice+CE, no DeepSupervisionWrapper.
        return DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)
