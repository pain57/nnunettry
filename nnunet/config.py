from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class NNUnetConfig:
    """Configuration for nnU-Net 3D pancreas segmentation."""

    # ---- Data ----
    # AMOS22: 15 organs + background = 16 classes
    # For pancreas-only: set to 2 and use label_id=10 filter
    num_classes: int = 16  # background + 15 abdominal organs
    modalities: int = 1    # CT only
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ---- Network Architecture ----
    base_features: int = 32
    num_stages: int = 5               # encoder stages (max downsampling 2^(num_stages-1))
    deep_supervision: bool = True
    deep_supervision_levels: int = 2  # how many decoder stages get auxiliary heads
    normalization: str = "instance"   # "instance" | "batch"
    activation: str = "leaky_relu"    # "leaky_relu" | "relu"

    # ---- Training ----
    batch_size: int = 2
    num_epochs: int = 1000
    initial_lr: float = 0.01
    weight_decay: float = 3e-5
    poly_lr_exponent: float = 0.9
    deep_supervision_weights: List[float] = field(default_factory=lambda: [1.0, 0.5])

    # ---- Loss ----
    dice_weight: float = 1.0
    ce_weight: float = 1.0

    # ---- Patch Sampling ----
    samples_per_volume: int = 8       # patches drawn per volume per epoch
    force_fg_ratio: float = 0.33      # fraction of patches that must contain foreground
    fg_min_ratio: float = 0.01        # minimum foreground fraction in a patch to count as "containing fg"

    # ---- Inference ----
    sliding_window_overlap: float = 0.5
    sliding_window_batch_size: int = 4
    gaussian_weight_sigma: float = 0.125  # sigma for Gaussian kernel (as fraction of patch size)

    # ---- Device ----
    device: str = "cuda"
