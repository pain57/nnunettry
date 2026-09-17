"""One module per step of the technical roadmap (all built on official nnU-Net v2).

Active experiments for AMOS22 MRI pancreas segmentation: 1 (baseline), 2
(ResEnc M/L), 3 (spacing), 6 (transformer backbones). Experiments 4 (ROI ->
duct) and 5 (clDice) are paused until duct ground truth is available.
"""

from . import exp1_baseline, exp2_backbone, exp3_spacing, exp6_models

EXPERIMENTS = {
    1: exp1_baseline,
    2: exp2_backbone,
    3: exp3_spacing,
    6: exp6_models,
}

__all__ = ["EXPERIMENTS"]
