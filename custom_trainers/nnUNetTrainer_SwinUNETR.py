"""Exp 6 — SwinUNETR (Hatamizadeh et al., MICCAI BrainLes 2022), via MONAI."""

import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_SwinUNETR(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision) -> nn.Module:
        from monai.networks.nets import SwinUNETR

        # SwinUNETR downsamples 4x -> img_size must be divisible by 32.
        img_size = tuple(arch_init_kwargs.get("img_size", (96, 96, 96)))

        return SwinUNETR(
            img_size=img_size,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
        )
