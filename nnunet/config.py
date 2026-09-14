"""Central configuration dataclass for the pancreas segmentation framework.

Every step of the technical roadmap maps to a field here:

    Exp1  baseline           -> backbone = "plain"
    Exp2  backbone           -> backbone = "resenc_m" / "resenc_l"
    Exp3  target spacing     -> target_spacing = (a, b, c)
    Exp4  class imbalance    -> use_roi = True / coarse_checkpoint
    Exp5  continuity         -> loss = "dice_ce_cldice" / cldice_weight
    Exp6  other architectures-> backbone = "unetr" / "swin_unetr" / "mednext_*"

A single :class:`NNUnetConfig` instance therefore fully describes a training
run, which keeps the experiment scripts short and reproducible.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NNUnetConfig:
    # ---- Data ----
    num_classes: int = 2                      # background + pancreas
    modalities: int = 1                       # CT only
    patch_size: Tuple[int, int, int] = (128, 128, 128)

    # ---- Resampling (Experiment 3: target spacing) ----
    # When set, every volume is resampled to this voxel spacing (mm) before
    # cropping / patch sampling. None keeps the native spacing.
    target_spacing: Optional[Tuple[float, float, float]] = None
    resample_order_image: int = 3             # cubic for images
    resample_order_label: int = 0             # nearest for labels

    # ---- Network architecture (Experiments 1, 2, 6) ----
    backbone: str = "plain"
    base_features: int = 32
    num_stages: int = 5
    deep_supervision: bool = True
    deep_supervision_levels: int = 2
    normalization: str = "instance"           # "instance" | "batch"
    activation: str = "leaky_relu"            # "leaky_relu" | "relu"

    # ResEnc presets (used only when backbone is "resenc_m"/"resenc_l")
    resenc_features_per_stage: Optional[Tuple[int, ...]] = None
    resenc_blocks_per_stage: Optional[Tuple[int, ...]] = None

    # UNETR presets (used only when backbone is "unetr")
    unetr_embed_dim: int = 768
    unetr_num_layers: int = 12
    unetr_num_heads: int = 12
    unetr_mlp_dim: int = 3072
    unetr_feature_size: int = 16
    unetr_patch_size: int = 16

    # SwinUNETR presets (used only when backbone is "swin_unetr")
    # window_size must divide every stage's spatial dims; for patch_size=2 and
    # a 128^3 patch the stage sizes are 64/32/16/8, so (8,8,8) is a safe choice.
    swin_embed_dim: int = 48
    swin_depths: Tuple[int, int, int, int] = (2, 2, 2, 2)
    swin_num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24)
    swin_window_size: Tuple[int, int, int] = (8, 8, 8)
    swin_patch_size: int = 2

    # MedNeXt presets (used only when backbone is "mednext_s"/"mednext_m")
    mednext_kernel_size: int = 5
    mednext_expansion: int = 4

    # ---- Training ----
    batch_size: int = 2
    num_epochs: int = 1000
    initial_lr: float = 0.01
    weight_decay: float = 3e-5
    poly_lr_exponent: float = 0.9
    deep_supervision_weights: List[float] = field(default_factory=lambda: [1.0, 0.5])
    val_every: int = 5
    save_every: int = 100

    # ---- Loss (Experiment 5: continuity) ----
    loss: str = "dice_ce"                     # "dice_ce" | "dice_ce_cldice"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    cldice_weight: float = 1.0
    cldice_iters: int = 3
    # Optional per-class CE weights (Experiment 4: extreme class imbalance)
    ce_class_weights: Optional[List[float]] = None

    # ---- Patch sampling ----
    samples_per_volume: int = 8
    force_fg_ratio: float = 0.33             # fraction of patches forced to contain fg
    fg_min_ratio: float = 0.01

    # ---- ROI / coarse-to-fine (Experiment 4) ----
    use_roi: bool = False                    # sample training patches inside the GT pancreas ROI
    roi_margin: int = 16
    coarse_checkpoint: Optional[str] = None  # coarse model for two-stage inference

    # ---- Inference ----
    sliding_window_overlap: float = 0.5
    sliding_window_batch_size: int = 4
    gaussian_weight_sigma: float = 0.125

    # ---- Device ----
    device: str = "cuda"
