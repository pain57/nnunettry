"""Weight initialization helpers shared by all backbones."""

import torch.nn as nn


def init_weights(module):
    """Kaiming init for convolutions, constant init for norm/embedding layers."""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
