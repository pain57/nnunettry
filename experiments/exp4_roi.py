"""Experiment 4 — ROI / coarse-to-fine.

    Solve extreme class imbalance.

The pancreas occupies only a tiny fraction of the abdomen, so a single-stage
model wastes most of its capacity on background. This experiment runs two stages:

  1. coarse model  — plain U-Net on the full volume (localize the pancreas);
  2. fine model    — same architecture, but trained with ``use_roi`` so patches
                     are drawn inside the ground-truth pancreas ROI.

At inference the two models are combined with
:class:`nnunet.inference.coarse_to_fine.CoarseToFinePredictor`.
"""

from nnunet.config import NNUnetConfig
from .common import run_training


def build_config(use_roi: bool, roi_margin: int = 16, **overrides) -> NNUnetConfig:
    cfg = NNUnetConfig()
    cfg.backbone = "plain"
    cfg.num_classes = 2
    cfg.loss = "dice_ce"
    cfg.num_epochs = 1000
    cfg.use_roi = use_roi
    cfg.roi_margin = roi_margin
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run(data_dir: str = "data", output_dir: str = "runs/exp4_roi",
        device: str = "cuda", roi_margin: int = 16, **overrides):
    coarse_out = f"{output_dir}/coarse"
    fine_out = f"{output_dir}/fine"

    print("\n--- Stage 1: coarse model (full volume) ---")
    run_training(build_config(use_roi=False, **overrides),
                 data_dir, coarse_out, device=device)

    print("\n--- Stage 2: fine model (ROI-focused sampling) ---")
    run_training(build_config(use_roi=True, roi_margin=roi_margin, **overrides),
                 data_dir, fine_out, device=device)

    return {"coarse": coarse_out, "fine": fine_out}
