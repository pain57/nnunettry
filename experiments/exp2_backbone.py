"""Exp 2 — residual-encoder (ResEnc) backbone comparison, M vs L.

In nnU-Net v2 the ResEnc architecture is chosen by the *planner*: planning with
``nnUNetPlannerResEncM`` / ``nnUNetPlannerResEncL`` writes the residual encoder
into the plans file, and the standard ``nnUNetTrainer`` then builds it. There is
no custom trainer here — this is the official mechanism.
"""

from .common import plan_and_preprocess, train

PLANNERS = {
    "M": "nnUNetPlannerResEncM",
    "L": "nnUNetPlannerResEncL",
}


def run(device=None, folds=None):
    plans = {}
    for size, planner in PLANNERS.items():
        plans_id = f"nnUNetPlansResEnc{size}"
        plan_and_preprocess(planner=planner, plans_identifier=plans_id)
        plans[plans_id] = planner

    for plans_id in plans:
        train("nnUNetTrainer", plans_identifier=plans_id, device=device, folds=folds)
