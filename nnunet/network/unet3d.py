"""Plain 3D U-Net (nnU-Net default backbone, Experiment 1 baseline)."""

import torch
import torch.nn as nn
from typing import List, Tuple
from .blocks import ConvBlock, UpsampleBlock, SegmentationHead
from .initialization import init_weights


class PlainConvUNet(nn.Module):
    """Encoder-decoder 3D U-Net with optional deep supervision.

    This is the default nnU-Net backbone: InstanceNorm + LeakyReLU, 5 stages,
    stride-2 downsampling in the encoder, trilinear upsampling in the decoder,
    and auxiliary heads attached to the lowest-resolution decoder outputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 32,
        num_stages: int = 5,
        deep_supervision: bool = True,
        deep_supervision_levels: int = 2,
        normalization: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.deep_supervision = deep_supervision
        self.deep_supervision_levels = min(deep_supervision_levels, num_stages - 1)

        # ---- Encoder ----
        self.encoder_stages = nn.ModuleList()
        encoder_features = []
        feats = base_features

        for stage in range(num_stages):
            in_ch = in_channels if stage == 0 else encoder_features[-1]
            out_ch = feats
            stride = 2 if stage > 0 else 1
            block = ConvBlock(in_ch, out_ch, stride=stride,
                              normalization=normalization, activation=activation)
            self.encoder_stages.append(block)
            encoder_features.append(out_ch)
            feats *= 2

        # ---- Bottleneck ----
        bottleneck_in = encoder_features[-1]
        bottleneck_out = feats // 2
        self.bottleneck = ConvBlock(bottleneck_in, bottleneck_out,
                                    normalization=normalization, activation=activation)

        # ---- Decoder ----
        self.decoder_stages = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        decoder_features = [bottleneck_out]

        for stage in range(num_stages - 1, -1, -1):
            skip_ch = encoder_features[stage]
            up_in = decoder_features[-1]
            upsample = UpsampleBlock(up_in, skip_ch, scale_factor=2,
                                     normalization=normalization, activation=activation)
            self.upsample_blocks.append(upsample)
            block = ConvBlock(skip_ch * 2, skip_ch,
                              normalization=normalization, activation=activation)
            self.decoder_stages.append(block)
            decoder_features.append(skip_ch)

        # ---- Output heads ----
        self.seg_head = SegmentationHead(decoder_features[-1], num_classes)

        self.ds_heads = nn.ModuleList()
        if self.deep_supervision:
            ds_indices = list(range(self.num_stages - self.deep_supervision_levels, self.num_stages))
            for idx in ds_indices:
                ds_ch = decoder_features[idx + 1]  # +1 because decoder_features starts with bottleneck
                self.ds_heads.append(SegmentationHead(ds_ch, num_classes))

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for enc in self.encoder_stages:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        ds_outputs = []
        ds_count = 0
        total_decoder_outputs = len(self.decoder_stages)

        for i, (dec, up) in enumerate(zip(self.decoder_stages, self.upsample_blocks)):
            skip = skips[self.num_stages - 1 - i]
            x = up(x, target_size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

            decoder_idx_from_end = total_decoder_outputs - 1 - i
            if self.deep_supervision and decoder_idx_from_end < self.deep_supervision_levels:
                ds_outputs.append(self.ds_heads[ds_count](x))
                ds_count += 1

        main_out = self.seg_head(x)

        if self.deep_supervision and ds_outputs:
            return main_out, ds_outputs
        return main_out


# Backward-compatible alias.
UNet3D = PlainConvUNet


def build_plain_unet(config) -> PlainConvUNet:
    """Build the plain nnU-Net from a NNUnetConfig instance."""
    return PlainConvUNet(
        in_channels=config.modalities,
        num_classes=config.num_classes,
        base_features=config.base_features,
        num_stages=config.num_stages,
        deep_supervision=config.deep_supervision,
        deep_supervision_levels=config.deep_supervision_levels,
        normalization=config.normalization,
        activation=config.activation,
    )


def build_3d_unet(config) -> PlainConvUNet:
    """Legacy alias kept for backward compatibility."""
    return build_plain_unet(config)
