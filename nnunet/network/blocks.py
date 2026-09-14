"""Shared building blocks for all network backbones."""

import torch
import torch.nn as nn
from typing import Tuple


def get_norm_layer(normalization: str, channels: int) -> nn.Module:
    """Return a normalization layer for a given channel count."""
    if normalization == "instance":
        return nn.InstanceNorm3d(channels)
    if normalization == "batch":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"Unknown normalization: {normalization}")


def get_activation(activation: str) -> nn.Module:
    """Return an activation module."""
    if activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation: {activation}")


class DropPath(nn.Module):
    """Stochastic depth (drop path) used by residual / ConvNeXt-style blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = mask.floor_()
        return x / keep_prob * mask


class ConvBlock(nn.Module):
    """Two stacked conv layers with normalization and activation.

    The first conv may use a stride (used for downsampling in the encoder).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        normalization: str = "instance",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = get_norm_layer(normalization, out_channels)
        self.act1 = get_activation(activation)
        self.drop1 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.norm2 = get_norm_layer(normalization, out_channels)
        self.act2 = get_activation(activation)
        self.drop2 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.norm1(self.conv1(x))))
        x = self.drop2(self.act2(self.norm2(self.conv2(x))))
        return x


class ResidualConvBlock(nn.Module):
    """ConvBlock with a residual skip (1x1 conv when channels/stride differ).

    This is the residual encoder block used by ResEncUNet (Experiment 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        normalization: str = "instance",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels, out_channels, stride=stride,
            normalization=normalization, activation=activation, dropout=dropout,
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.skip(x)


class UpsampleBlock(nn.Module):
    """Upsampling via trilinear interpolate + 1x1 conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = "trilinear",
        normalization: str = "instance",
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            get_norm_layer(normalization, out_channels),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=target_size, mode=self.mode, align_corners=False)
        return self.conv(x)


class SegmentationHead(nn.Module):
    """1x1x1 conv producing per-class logits."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
