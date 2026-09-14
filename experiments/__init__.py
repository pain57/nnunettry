"""Experiment scripts, one module per step of the technical roadmap."""

from . import (
    exp1_baseline,
    exp2_backbone,
    exp3_spacing,
    exp4_roi,
    exp5_cldice,
    exp6_models,
)

EXPERIMENTS = {
    1: exp1_baseline,
    2: exp2_backbone,
    3: exp3_spacing,
    4: exp4_roi,
    5: exp5_cldice,
    6: exp6_models,
}

__all__ = ["EXPERIMENTS"]
