"""Network backbones and the ``build_network`` factory.

The backbone registry maps the ``config.backbone`` string to a builder. This is
the single switch used by Experiments 1, 2 and 6:

    plain       -> PlainConvUNet (nnU-Net default, baseline)
    resenc_m    -> ResEncUNet M   (residual encoder)
    resenc_l    -> ResEncUNet L   (residual encoder, larger)
    unetr       -> UNETR          (ViT encoder)
    swin_unetr  -> SwinUNETR      (Swin Transformer encoder)
    mednext_s   -> MedNeXt S      (ConvNeXt encoder-decoder)
    mednext_m   -> MedNeXt M      (ConvNeXt encoder-decoder)
"""

import torch.nn as nn
from .unet3d import PlainConvUNet, build_plain_unet, build_3d_unet
from .resenc import ResEncUNet, build_resenc_unet
from .unetr import UNETR, build_unetr
from .swin_unetr import SwinUNETR, build_swin_unetr
from .mednext import MedNeXt, build_mednext

BACKBONE_CHOICES = (
    "plain",
    "resenc_m",
    "resenc_l",
    "unetr",
    "swin_unetr",
    "mednext_s",
    "mednext_m",
)

_BUILDERS = {
    "plain": build_plain_unet,
    "resenc_m": lambda cfg: build_resenc_unet(cfg, size="m"),
    "resenc_l": lambda cfg: build_resenc_unet(cfg, size="l"),
    "unetr": build_unetr,
    "swin_unetr": build_swin_unetr,
    "mednext_s": lambda cfg: build_mednext(cfg, size="s"),
    "mednext_m": lambda cfg: build_mednext(cfg, size="m"),
}


def build_network(config) -> nn.Module:
    """Build a segmentation network from a NNUnetConfig instance."""
    if config.backbone not in _BUILDERS:
        raise ValueError(
            f"Unknown backbone '{config.backbone}'. Choose from {BACKBONE_CHOICES}"
        )
    return _BUILDERS[config.backbone](config)


__all__ = [
    "PlainConvUNet",
    "ResEncUNet",
    "UNETR",
    "SwinUNETR",
    "MedNeXt",
    "build_network",
    "build_3d_unet",
    "BACKBONE_CHOICES",
]
