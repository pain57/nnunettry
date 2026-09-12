import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Two conv layers with normalization and activation. First conv can use stride."""

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
        norm_layer = nn.InstanceNorm3d if normalization == "instance" else nn.BatchNorm3d
        act_layer = nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.act1 = act_layer
        self.drop1 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.norm2 = norm_layer(out_channels)
        self.act2 = act_layer
        self.drop2 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x


class ResidualConvBlock(nn.Module):
    """ConvBlock with residual connection (1x1 conv when channels differ)."""

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
    """Upsampling via interpolate + conv, or transposed conv."""

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
            nn.InstanceNorm3d(out_channels) if normalization == "instance" else nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=target_size, mode=self.mode, align_corners=False)
        x = self.conv(x)
        return x


class SegmentationHead(nn.Module):
    """1x1x1 conv to produce segmentation logits."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
