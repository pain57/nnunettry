import torch.nn as nn


def init_weights(module):
    """Apply Kaiming normal initialization to Conv3d and ConvTranspose3d layers."""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d)):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
