"""Experiment 6 — compare against other architectures.

    UNETR / SwinUNETR / MedNeXt vs the nnU-Net baseline.

Trains a set of modern transformer / ConvNeXt backbones and prints a parameter /
validation-Dice comparison. The UNETR defaults are reduced (embed dim 384, 6
layers) to stay tractable on a single GPU; override them for full-scale runs.
"""

from nnunet.config import NNUnetConfig
from .common import run_training

MODELS = {
    "plain": {},
    "resenc_m": {},
    "unetr": {
        "unetr_embed_dim": 384,
        "unetr_num_layers": 6,
        "unetr_num_heads": 6,
        "unetr_mlp_dim": 1536,
    },
    "swin_unetr": {},
    "mednext_s": {},
    "mednext_m": {},
}


def build_config(backbone: str, **overrides) -> NNUnetConfig:
    cfg = NNUnetConfig()
    cfg.backbone = backbone
    cfg.num_classes = 2
    cfg.loss = "dice_ce"
    cfg.num_epochs = 1000
    # Transformer / ConvNeXt backbones emit a single output (no deep supervision).
    if backbone not in ("plain", "resenc_m", "resenc_l"):
        cfg.deep_supervision = False
    for k, v in MODELS[backbone].items():
        setattr(cfg, k, v)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(data_dir: str = "data", output_dir: str = "runs/exp6_models",
        device: str = "cuda", models=None, **overrides):
    models = models if models is not None else MODELS.keys()
    results = {}
    for backbone in models:
        cfg = build_config(backbone, **overrides)
        out = f"{output_dir}/{backbone}"
        results[backbone] = run_training(cfg, data_dir, out, device=device)

    print("\n=== Experiment 6 comparison ===")
    for backbone, trainer in results.items():
        print(f"  {backbone:12s} best val Dice: {trainer.best_val_dice:.4f}")
    return results
