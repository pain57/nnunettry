"""UNETR (Hatamizadeh et al., 2022) — Experiment 6 comparison model.

A Vision Transformer encoder operates on 16^3 patches of the 3D volume and its
output is decoded by a small convolutional decoder that upsamples back to full
resolution. Skip connections come from a lightweight CNN encoder (MONAI-style),
which is more stable than attaching the decoder to intermediate ViT layers.
"""

import torch
import torch.nn as nn
from typing import Tuple
from .blocks import get_norm_layer, get_activation
from .initialization import init_weights


class PatchEmbedding3D(nn.Module):
    """Linear projection of non-overlapping 3D patches into token embeddings."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, embed_dim, D/P, H/P, W/P) -> (B, N, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvBasicBlock(nn.Module):
    """Two conv layers with stride, used by the CNN skip encoder."""

    def __init__(self, in_ch, out_ch, stride=1, normalization="instance", activation="leaky_relu"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            get_norm_layer(normalization, out_ch),
            get_activation(activation),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            get_norm_layer(normalization, out_ch),
            get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class UNETRUpBlock(nn.Module):
    """Transposed-conv upsampling followed by concat with a skip and two convs."""

    def __init__(self, in_ch, out_ch, normalization="instance", activation="leaky_relu"):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            get_norm_layer(normalization, out_ch),
            get_activation(activation),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            get_norm_layer(normalization, out_ch),
            get_activation(activation),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNETR(nn.Module):
    """UNETR: ViT encoder + CNN skip encoder + convolutional decoder."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        feature_size: int = 16,
        normalization: str = "instance",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = tuple(s // patch_size for s in img_size)
        self.num_tokens = self.grid[0] * self.grid[1] * self.grid[2]

        # ---- ViT encoder ----
        self.patch_embedding = PatchEmbedding3D(in_channels, embed_dim, patch_size)
        self.position_encoding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- CNN skip encoder (features at 1/1, 1/2, 1/4, 1/8) ----
        f = feature_size
        self.encoder1 = ConvBasicBlock(in_channels, f, stride=1, normalization=normalization, activation=activation)
        self.encoder2 = ConvBasicBlock(f, f * 2, stride=2, normalization=normalization, activation=activation)
        self.encoder3 = ConvBasicBlock(f * 2, f * 4, stride=2, normalization=normalization, activation=activation)
        self.encoder4 = ConvBasicBlock(f * 4, f * 8, stride=2, normalization=normalization, activation=activation)

        # ---- Decoder ----
        self.decoder5 = UNETRUpBlock(embed_dim, f * 8, normalization, activation)
        self.decoder4 = UNETRUpBlock(f * 8, f * 4, normalization, activation)
        self.decoder3 = UNETRUpBlock(f * 4, f * 2, normalization, activation)
        self.decoder2 = UNETRUpBlock(f * 2, f, normalization, activation)
        self.out = nn.Conv3d(f, out_channels, kernel_size=1)

        self.apply(init_weights)

    def proj_view(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, embed_dim) -> (B, embed_dim, d, h, w)
        B, _, C = x.shape
        d, h, w = self.grid
        return x.transpose(1, 2).view(B, C, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN skip encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # ViT encoder
        tokens = self.patch_embedding(x)
        tokens = self.dropout(tokens + self.position_encoding[:, :tokens.shape[1]])
        tokens = self.transformer(tokens)
        x = self.proj_view(tokens)

        # Decoder with skips
        x = self.decoder5(x, enc4)
        x = self.decoder4(x, enc3)
        x = self.decoder3(x, enc2)
        x = self.decoder2(x, enc1)
        return self.out(x)


def build_unetr(config) -> UNETR:
    """Build a UNETR from a NNUnetConfig instance."""
    return UNETR(
        in_channels=config.modalities,
        out_channels=config.num_classes,
        img_size=config.patch_size,
        patch_size=config.unetr_patch_size,
        embed_dim=config.unetr_embed_dim,
        num_layers=config.unetr_num_layers,
        num_heads=config.unetr_num_heads,
        mlp_dim=config.unetr_mlp_dim,
        feature_size=config.unetr_feature_size,
        normalization=config.normalization,
        activation=config.activation,
    )
