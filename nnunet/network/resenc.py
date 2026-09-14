"""Residual-encoder U-Net (nnU-Net v2 ResEnc family, Experiment 2).

ResEncUNet keeps the plain convolutional decoder of nnU-Net but replaces the
encoder with residual blocks, following the nnU-Net v2 ``ResEncM`` / ``ResEncL``
presets. The residual blocks are defined in :mod:`nnunet.network.blocks` and the
encoder stage counts / channel depths are fully configurable so M/L size
variants only differ by two tuples.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .blocks import ConvBlock, ResidualConvBlock, UpsampleBlock, SegmentationHead
from .initialization import init_weights

# Adapted from the nnU-Net v2 ResEnc presets. The exact channel/block counts are
# documented approximations of the upstream defaults.
RESENC_M_FEATURES: Tuple[int, ...] = (32, 64, 128, 256, 320, 320)
RESENC_M_BLOCKS: Tuple[int, ...] = (1, 3, 4, 6, 6, 6)

RESENC_L_FEATURES: Tuple[int, ...] = (32, 64, 128, 256, 512, 512, 512)
RESENC_L_BLOCKS: Tuple[int, ...] = (1, 3, 4, 6, 6, 6, 6)

RESENC_PRESETS = {
    "m": (RESENC_M_FEATURES, RESENC_M_BLOCKS),
    "l": (RESENC_L_FEATURES, RESENC_L_BLOCKS),
}


class ResEncUNet(nn.Module):
    """3D U-Net with a residual encoder and a plain convolutional decoder.

    Args:
        input_channels: Number of input modalities (1 for CT).
        output_channels: Number of segmentation classes.
        features_per_stage: Channels per encoder stage, deepest last.
        n_blocks_per_stage: Residual blocks per encoder stage.
        deep_supervision: Enable auxiliary heads on the decoder.
        deep_supervision_levels: Number of auxiliary heads.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 2,
        features_per_stage: Tuple[int, ...] = RESENC_M_FEATURES,
        n_blocks_per_stage: Tuple[int, ...] = RESENC_M_BLOCKS,
        kernel_size: int = 3,
        deep_supervision: bool = True,
        deep_supervision_levels: int = 2,
        normalization: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()
        assert len(features_per_stage) == len(n_blocks_per_stage), \
            "features_per_stage and n_blocks_per_stage must have equal length"
        self.num_stages = len(features_per_stage)
        self.deep_supervision = deep_supervision
        self.deep_supervision_levels = min(deep_supervision_levels, self.num_stages - 1)

        # ---- Residual encoder ----
        self.encoder = nn.ModuleList()
        for s in range(self.num_stages):
            in_ch = input_channels if s == 0 else features_per_stage[s - 1]
            out_ch = features_per_stage[s]
            stride = 2 if s > 0 else 1
            blocks = []
            for b in range(n_blocks_per_stage[s]):
                b_stride = stride if b == 0 else 1
                b_in = in_ch if b == 0 else out_ch
                blocks.append(ResidualConvBlock(
                    b_in, out_ch, stride=b_stride,
                    normalization=normalization, activation=activation,
                ))
            self.encoder.append(nn.Sequential(*blocks))

        # ---- Plain decoder (mirrors the nnU-Net decoder) ----
        self.upsample_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for s in range(self.num_stages - 1, 0, -1):
            up_in = features_per_stage[s]
            up_out = features_per_stage[s - 1]
            self.upsample_blocks.append(UpsampleBlock(
                up_in, up_out, scale_factor=2,
                normalization=normalization, activation=activation,
            ))
            self.decoder.append(ConvBlock(
                up_out * 2, up_out, kernel_size=kernel_size,
                normalization=normalization, activation=activation,
            ))

        # ---- Output heads ----
        self.seg_head = SegmentationHead(features_per_stage[0], output_channels)
        self.ds_heads = nn.ModuleList()
        if self.deep_supervision:
            for i in range(self.deep_supervision_levels):
                ds_ch = features_per_stage[self.num_stages - 2 - i]
                self.ds_heads.append(SegmentationHead(ds_ch, output_channels))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)  # skips[0] full-res ... skips[-1] deepest

        ds_outputs = []
        for i, (up, dec) in enumerate(zip(self.upsample_blocks, self.decoder)):
            skip = skips[self.num_stages - 2 - i]
            x = up(x, target_size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            if self.deep_supervision and i < self.deep_supervision_levels:
                ds_outputs.append(self.ds_heads[i](x))

        main_out = self.seg_head(x)
        if self.deep_supervision and ds_outputs:
            return main_out, ds_outputs
        return main_out


def build_resenc_unet(config, size: str = "m") -> ResEncUNet:
    """Build a ResEncUNet from a config, using an M/L preset unless overridden."""
    features, blocks = RESENC_PRESETS[size]
    if config.resenc_features_per_stage is not None:
        features = config.resenc_features_per_stage
    if config.resenc_blocks_per_stage is not None:
        blocks = config.resenc_blocks_per_stage

    return ResEncUNet(
        input_channels=config.modalities,
        output_channels=config.num_classes,
        features_per_stage=tuple(features),
        n_blocks_per_stage=tuple(blocks),
        deep_supervision=config.deep_supervision,
        deep_supervision_levels=config.deep_supervision_levels,
        normalization=config.normalization,
        activation=config.activation,
    )
