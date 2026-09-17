"""Exp 5 — topology-aware clDice loss (Dice+CE vs Dice+CE+clDice).  *** PAUSED ***

Not a primary experiment for single-organ pancreas segmentation (clDice matters
for thin tubes). Kept for the future pancreatic-duct / bile-duct work;
``run_experiment.py`` does not map ``--exp 5``.
"""


def run(device=None, folds=None):
    raise NotImplementedError(
        "Exp 5 (clDice) is paused for single-organ pancreas segmentation. "
        "Re-enable for the future pancreatic-duct / bile-duct task."
    )
