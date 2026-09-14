"""MedNeXt (Roy et al., 2023) — Experiment 6 comparison model.

A fully ConvNeXt-style encoder-decoder for 3D medical volumes: depthwise
large-kernel convolutions, expansion ratio 4, GroupNorm(=LayerNorm) and
residual connections. The S/M presets approximate the paper's configurations
using a configurable kernel size (3x3x3 or 5x5x5).
"""

import torch
import torch.nn as nn
from typing import Tuple
from .blocks import DropPath, SegmentationHead
from .initialization import init_weights


class MedNeXtBlock(nn.Module):
    """ConvNeXt-style block: DW conv with expansion, optional residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5,
                 expansion: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.use_residual = (in_ch == out_ch)
        mid = in_ch * expansion
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, mid, kernel_size=1)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv3d(mid, mid, kernel_size, padding=kernel_size // 2, groups=mid)
        self.norm2 = nn.GroupNorm(1, mid)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv3d(mid, out_ch, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(self.act1(self.conv1(self.norm1(x))))
        x = self.conv2(self.act2(self.norm2(x)))
        x = self.drop_path(x)
        if self.use_residual:
            x = x + shortcut
        return x


class MedNeXtDownBlock(nn.Module):
    """Strided ConvNeXt block for encoder downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, expansion: int = 4):
        super().__init__()
        mid = in_ch * expansion
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, mid, kernel_size=1)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv3d(mid, mid, kernel_size, stride=2,
                                padding=kernel_size // 2, groups=mid)
        self.norm2 = nn.GroupNorm(1, mid)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv3d(mid, out_ch, kernel_size=1)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        x = self.dwconv(self.act1(self.conv1(self.norm1(x))))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + shortcut


class MedNeXtUpBlock(nn.Module):
    """Transposed-conv upsampling + skip concat + ConvNeXt block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel_size: int = 5, expansion: int = 4):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = MedNeXtBlock(out_ch + skip_ch, out_ch, kernel_size, expansion)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# Adapted approximations of the MedNeXt S/M/L configurations.
MEDNEXT_S_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)
MEDNEXT_S_BLOCKS: Tuple[int, ...] = (2, 2, 2, 2)

MEDNEXT_M_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)
MEDNEXT_M_BLOCKS: Tuple[int, ...] = (3, 4, 4, 3)

MEDNEXT_PRESETS = {
    "s": (MEDNEXT_S_CHANNELS, MEDNEXT_S_BLOCKS),
    "m": (MEDNEXT_M_CHANNELS, MEDNEXT_M_BLOCKS),
}


class MedNeXt(nn.Module):
    """MedNeXt encoder-decoder with ConvNeXt blocks and skip connections."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        kernel_size: int = 5,
        expansion: int = 4,
        channels: Tuple[int, ...] = MEDNEXT_M_CHANNELS,
        blocks_per_stage: Tuple[int, ...] = MEDNEXT_M_BLOCKS,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert len(channels) == len(blocks_per_stage)
        self.num_stages = len(channels)

        self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=1)

        # ---- Encoder ----
        self.encoder = nn.ModuleList()
        for i in range(self.num_stages):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            layers = []
            if i > 0:
                layers.append(MedNeXtDownBlock(in_ch, out_ch, kernel_size, expansion))
            n_residual = blocks_per_stage[i] - (1 if i > 0 else 0)
            for _ in range(n_residual):
                layers.append(MedNeXtBlock(out_ch, out_ch, kernel_size, expansion, drop_path))
            self.encoder.append(nn.Sequential(*layers))

        # ---- Decoder ----
        self.decoder = nn.ModuleList()
        for i in range(self.num_stages - 1, 0, -1):
            self.decoder.append(MedNeXtUpBlock(
                channels[i], channels[i - 1], channels[i - 1], kernel_size, expansion,
            ))

        self.seg_head = SegmentationHead(channels[0], out_channels)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)

        for i, dec in enumerate(self.decoder):
            skip = skips[self.num_stages - 2 - i]
            x = dec(x, skip)

        return self.seg_head(x)


def build_mednext(config, size: str = "m") -> MedNeXt:
    """Build a MedNeXt from a config, using an S/M preset."""
    channels, blocks = MEDNEXT_PRESETS[size]
    return MedNeXt(
        in_channels=config.modalities,
        out_channels=config.num_classes,
        kernel_size=config.mednext_kernel_size,
        expansion=config.mednext_expansion,
        channels=tuple(channels),
        blocks_per_stage=tuple(blocks),
    )
