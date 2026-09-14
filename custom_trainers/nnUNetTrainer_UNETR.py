"""Exp 6 — UNETR (Hatamizadeh et al., WACV 2022), via the official MONAI model."""

import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_UNETR(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision) -> nn.Module:
        from monai.networks.nets import UNETR

        # MONAI UNETR needs a fixed input size equal to the nnU-Net patch size.
        # ``img_size`` is injected into the plans file by
        # ``nnunet_utils.plans_patch.patch_arch_init_kwargs`` (see exp6_models.py).
        img_size = tuple(arch_init_kwargs.get("img_size", (96, 96, 96)))

        # UNETR has no deep-supervision heads -> single logit output, which the
        # official DeepSupervisionWrapper handles transparently.
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
