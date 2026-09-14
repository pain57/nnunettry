"""Exp 6 — MedNeXt (Roy et al., MICCAI 2023), via the official MedNeXt package."""

import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_MedNeXt(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision) -> nn.Module:
        try:
            from MedNeXt import MedNeXt
        except ImportError as e:
            raise ImportError(
                "MedNeXt is not installed. Install it from source:\n"
                "    pip install git+https://github.com/MIC-DKFZ/MedNeXt.git"
            ) from e

        # deep_supervision=False keeps a single logit output, which the official
        # DeepSupervisionWrapper handles transparently.
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
