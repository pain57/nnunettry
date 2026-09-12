import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .blocks import ConvBlock, UpsampleBlock, SegmentationHead
from .initialization import init_weights


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric biomedical image segmentation.

    Encoder-decoder with skip connections and optional deep supervision.
    Follows the nnU-Net design: InstanceNorm, LeakyReLU, 5 stages,
    stride-2 downsampling in the encoder, and deep supervision from
    the lower-resolution decoder stages.

    Args:
        in_channels: Number of input modalities (1 for CT).
        num_classes: Number of segmentation classes (2 for bg+pancreas).
        base_features: Features in the first encoder stage (default 32).
        num_stages: Number of encoder/decoder stages (default 5).
        deep_supervision: Enable auxiliary segmentation heads.
        deep_supervision_levels: How many decoder stages get aux heads (1..num_stages-1).
        normalization: "instance" or "batch".
        activation: "leaky_relu" or "relu".
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
            stride = 2 if stage > 0 else 1  # no striding at first stage
            block = ConvBlock(in_ch, out_ch, stride=stride,
                              normalization=normalization, activation=activation)
            self.encoder_stages.append(block)
            encoder_features.append(out_ch)
            feats *= 2  # double features at each stage

        # ---- Bottleneck ----
        bottleneck_in = encoder_features[-1]
        bottleneck_out = feats // 2  # match last encoder output channels * 2
        self.bottleneck = ConvBlock(bottleneck_in, bottleneck_out,
                                    normalization=normalization, activation=activation)

        # ---- Decoder ----
        self.decoder_stages = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        decoder_features = [bottleneck_out]

        for stage in range(num_stages - 1, -1, -1):
            skip_ch = encoder_features[stage]
            up_in = decoder_features[-1]

            if stage > 0:
                upsample = UpsampleBlock(up_in, skip_ch, scale_factor=2,
                                         normalization=normalization, activation=activation)
                self.upsample_blocks.append(upsample)
                # After concat with skip: skip_ch (skip) + skip_ch (upsampled) = 2 * skip_ch
                decode_ch = skip_ch
                block = ConvBlock(skip_ch * 2, decode_ch,
                                  normalization=normalization, activation=activation)
            else:
                upsample = UpsampleBlock(up_in, skip_ch, scale_factor=2,
                                         normalization=normalization, activation=activation)
                self.upsample_blocks.append(upsample)
                decode_ch = skip_ch
                block = ConvBlock(skip_ch * 2, decode_ch,
                                  normalization=normalization, activation=activation)

            self.decoder_stages.append(block)
            decoder_features.append(decode_ch)

        # ---- Output Heads ----
        # Main segmentation head at full resolution (last decoder output)
        self.seg_head = SegmentationHead(decoder_features[-1], num_classes)

        # Deep supervision heads: attach to lower-resolution decoder outputs
        self.ds_heads = nn.ModuleList()
        if self.deep_supervision:
            # decoder_features indices for deep supervision:
            # 0=bottleneck_out, 1=last_decoder_up, ..., num_stages=final_decoder
            # We attach heads to intermediate decoder outputs (before final)
            ds_indices = list(range(self.num_stages - self.deep_supervision_levels, self.num_stages))
            for idx in ds_indices:
                # idx points to decoder_features entry (after upsample+skip+conv)
                ds_ch = decoder_features[idx + 1]  # +1 because decoder_features starts with bottleneck
                head = SegmentationHead(ds_ch, num_classes)
                self.ds_heads.append(head)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, D, H, W) input tensor.
        Returns:
            If deep_supervision=False: (B, num_classes, D, H, W) logits.
            If deep_supervision=True: (main_logits, [ds_logits_1, ds_logits_2, ...]).
        """
        # ---- Encoder ----
        skips = []
        for enc in self.encoder_stages:
            x = enc(x)
            skips.append(x)

        # ---- Bottleneck ----
        x = self.bottleneck(x)

        # ---- Decoder ----
        ds_outputs = []
        ds_count = 0
        total_decoder_outputs = len(self.decoder_stages)

        for i, (dec, up) in enumerate(zip(self.decoder_stages, self.upsample_blocks)):
            skip = skips[self.num_stages - 1 - i]
            x = up(x, target_size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

            # Deep supervision: collect outputs from specified decoder stages
            # (counting from the deepest/lowest-resolution decoder output)
            decoder_idx_from_end = total_decoder_outputs - 1 - i
            if self.deep_supervision and decoder_idx_from_end < self.deep_supervision_levels:
                ds_outputs.append(self.ds_heads[ds_count](x))
                ds_count += 1

        # ---- Main Output ----
        main_out = self.seg_head(x)

        if self.deep_supervision and ds_outputs:
            return main_out, ds_outputs
        return main_out


def build_3d_unet(config) -> UNet3D:
    """Factory function: build a UNet3D from a NNUnetConfig instance."""
    return UNet3D(
        in_channels=config.modalities,
        num_classes=config.num_classes,
        base_features=config.base_features,
        num_stages=config.num_stages,
        deep_supervision=config.deep_supervision,
        deep_supervision_levels=config.deep_supervision_levels,
        normalization=config.normalization,
        activation=config.activation,
    )
