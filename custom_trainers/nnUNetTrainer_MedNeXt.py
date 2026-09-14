"""Exp 6 — MedNeXt (official MIC-DKFZ package) trainer.

MedNeXt is fully convolutional (no fixed img_size) and single-output, so only
the network construction and the no-deep-supervision loss are overridden.

Requires: ``pip install git+https://github.com/MIC-DKFZ/MedNeXt.git``
"""

import torch.nn as nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_MedNeXt(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision) -> nn.Module:
        from MedNeXt import MedNeXt
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3],
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2],
        )

    def _build_loss(self):
        # No deep supervision: plain Dice+CE, no DeepSupervisionWrapper.
        return DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)
