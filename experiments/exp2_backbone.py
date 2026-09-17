"""Exp 2 — residual-encoder (ResEnc) backbone comparison, M vs L (pancreas).

In nnU-Net v2 the ResEnc architecture is chosen by the *planner*: planning with
``nnUNetPlannerResEncM`` / ``nnUNetPlannerResEncL`` writes the residual encoder
into a plans file named ``nnUNetResEncUNetMPlans`` / ``nnUNetResEncUNetLPlans``,
and the standard ``nnUNetTrainer`` then builds it. No custom trainer here — this
is the official mechanism. The planner flag is ``-pl`` (not ``-planner``).
"""

from .common import plan_and_preprocess, train

PLANNERS = {
    "nnUNetResEncUNetMPlans": "nnUNetPlannerResEncM",
    "nnUNetResEncUNetLPlans": "nnUNetPlannerResEncL",
}


def run(device=None, folds=None):
    for plans_id, planner in PLANNERS.items():
        plan_and_preprocess(planner=planner, plans_identifier=plans_id)
        train("nnUNetTrainer", plans_identifier=plans_id, device=device, folds=folds)
