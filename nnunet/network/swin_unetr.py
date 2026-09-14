"""SwinUNETR (Hatamizadeh et al., 2022) — Experiment 6 comparison model.

A 3D Swin Transformer encoder (window attention with shifted windows) is paired
with a transposed-convolution decoder that upsamples back to full resolution
using skip connections from each encoder stage. This is a compact reimplementation
of the BraTS-winning architecture; window / depth / head counts are configurable.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from .initialization import init_weights


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """(B, D, H, W, C) -> (B * nW, wD*wH*wW, C)."""
    B, D, H, W, C = x.shape
    wD, wH, wW = window_size
    x = x.view(B, D // wD, wD, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wD * wH * wW, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int],
                   D: int, H: int, W: int) -> torch.Tensor:
    """(B * nW, wD*wH*wW, C) -> (B, D, H, W, C)."""
    wD, wH, wW = window_size
    B = int(windows.shape[0] / ((D // wD) * (H // wH) * (W // wW)))
    x = windows.view(B, D // wD, H // wH, W // wW, wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """Multi-head self-attention within a window, with relative position bias."""

    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        wD, wH, wW = window_size
        table_len = (2 * wD - 1) * (2 * wH - 1) * (2 * wW - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_len, num_heads))

        coords = torch.stack(torch.meshgrid(
            torch.arange(wD), torch.arange(wH), torch.arange(wW), indexing="ij"
        ))
        coords_flatten = coords.reshape(3, -1)  # (3, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += wD - 1
        relative_coords[:, :, 1] += wH - 1
        relative_coords[:, :, 2] += wW - 1
        relative_coords[:, :, 0] *= (2 * wH - 1) * (2 * wW - 1)
        relative_coords[:, :, 1] *= (2 * wW - 1)
        self.relative_position_index = relative_coords.sum(-1)  # (N, N)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(N, N, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = attn + rel_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


def _shift_slices(size: int, window: int, shift: int):
    """Helper producing the three cyclic-shift slices for one axis."""
    return (slice(0, -window), slice(-window, -shift), slice(-shift, None))


class SwinTransformerBlock3D(nn.Module):
    """One Swin block: (shifted) window attention followed by an MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int],
                 shift_size: Tuple[int, int, int], mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        wD, wH, wW = self.window_size
        shift = self.shift_size
        do_shift = any(s > 0 for s in shift)

        if do_shift:
            shifted = torch.roll(x, shifts=(-shift[0], -shift[1], -shift[2]), dims=(1, 2, 3))
            attn_mask = self._compute_mask(D, H, W)
        else:
            shifted = x
            attn_mask = None

        x_windows = window_partition(shifted, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted = window_reverse(attn_windows, self.window_size, D, H, W)

        if do_shift:
            x = torch.roll(shifted, shifts=shift, dims=(1, 2, 3))
        else:
            x = shifted

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_mask(self, D: int, H: int, W: int) -> torch.Tensor:
        """Return the attention mask (nW, N, N) for a shifted-window layer."""
        wD, wH, wW = self.window_size
        shift = self.shift_size
        img_mask = torch.zeros((1, D, H, W, 1))
        cnt = 0
        for d in _shift_slices(D, wD, shift[0]):
            for h in _shift_slices(H, wH, shift[1]):
                for w in _shift_slices(W, wW, shift[2]):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # (nW, N, 1)
        mask_windows = mask_windows.view(-1, wD * wH * wW)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        return attn_mask


class PatchEmbed3D(nn.Module):
    """Stem: Conv3d patch projection into the first-stage feature grid."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, D, H, W, embed_dim)
        x = self.proj(x).permute(0, 2, 3, 4, 1)
        return self.norm(x)


class PatchMerging3D(nn.Module):
    """Downsample 2x by merging 2x2x2 patches and projecting to 2x channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        x = x.view(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, D // 2, H // 2, W // 2, 8 * C)
        return self.reduction(self.norm(x))


class BasicLayer3D(nn.Module):
    """One Swin stage: optional patch merging + a stack of transformer blocks."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: Tuple[int, int, int], mlp_ratio: float,
                 downsample: bool, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0, 0, 0) if i % 2 == 0 else (w // 2 for w in window_size)
            self.blocks.append(SwinTransformerBlock3D(
                dim, num_heads, window_size, tuple(shift), mlp_ratio, dropout,
            ))
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinUpBlock(nn.Module):
    """Transposed-conv upsampling + skip concat + two convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNETR(nn.Module):
    """Swin Transformer 3D encoder + convolutional decoder."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        embed_dim: int = 48,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)

        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(BasicLayer3D(
                dims[i], depths[i], num_heads[i], window_size, mlp_ratio,
                downsample=(i < 3), dropout=dropout,
            ))

        # Decoder: upsample back to full resolution with skip connections from
        # the encoder stages. Resolutions are 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1.
        self.decoder = nn.ModuleList()
        self.decoder.append(SwinUpBlock(dims[3], dims[2], dims[2]))  # 1/16 -> 1/8 + s2
        self.decoder.append(SwinUpBlock(dims[2], dims[1], dims[1]))  # 1/8  -> 1/4 + s1
        self.decoder.append(SwinUpBlock(dims[1], dims[0], dims[0]))  # 1/4  -> 1/2 + s0

        # Final upsampling to full resolution has no encoder skip (the stem
        # already downsamples by patch_size), so use a plain transposed conv.
        self.up_final = nn.ConvTranspose3d(dims[0], dims[0], kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.InstanceNorm3d(dims[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dims[0], dims[0], kernel_size=3, padding=1, bias=False),
        )
        self.out = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)        # 1/patch (1/2)
        skips = [x]                    # capture the stem output as the first skip
        for layer in self.layers:
            x = layer(x)
            skips.append(x)            # 1/4, 1/8, 1/16, 1/16

        # skips are (B, D, H, W, C) -> (B, C, D, H, W)
        s0 = skips[0].permute(0, 4, 1, 2, 3)   # 1/2
        s1 = skips[1].permute(0, 4, 1, 2, 3)   # 1/4
        s2 = skips[2].permute(0, 4, 1, 2, 3)   # 1/8
        s3 = skips[3].permute(0, 4, 1, 2, 3)   # 1/16 (deepest)

        x = self.decoder[0](s3, s2)   # 1/16 -> 1/8
        x = self.decoder[1](x, s1)    # 1/8  -> 1/4
        x = self.decoder[2](x, s0)    # 1/4  -> 1/2
        x = self.final_conv(self.up_final(x))  # 1/2 -> 1/1
        return self.out(x)


def build_swin_unetr(config) -> SwinUNETR:
    """Build a SwinUNETR from a NNUnetConfig instance."""
    return SwinUNETR(
        in_channels=config.modalities,
        out_channels=config.num_classes,
        embed_dim=config.swin_embed_dim,
        depths=config.swin_depths,
        num_heads=config.swin_num_heads,
        window_size=config.swin_window_size,
        patch_size=config.swin_patch_size,
    )
