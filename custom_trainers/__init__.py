"""Custom nnU-Net v2 trainers (Experiments 5 and 6).

Each trainer subclasses the official ``nnUNetTrainer`` and only overrides the
documented extension points (``_build_loss`` / ``build_network_architecture``).

To make them resolvable by name from ``nnUNetv2_train`` / ``nnUNetv2_predict``
(which look trainers up inside the installed ``nnunetv2`` package), run once:

    python install_custom_trainers.py
"""

from .nnUNetTrainer_DiceCEclDice import nnUNetTrainer_DiceCEclDice
from .nnUNetTrainer_UNETR import nnUNetTrainer_UNETR
from .nnUNetTrainer_SwinUNETR import nnUNetTrainer_SwinUNETR
from .nnUNetTrainer_MedNeXt import nnUNetTrainer_MedNeXt

__all__ = [
    "nnUNetTrainer_DiceCEclDice",
    "nnUNetTrainer_UNETR",
    "nnUNetTrainer_SwinUNETR",
    "nnUNetTrainer_MedNeXt",
]
