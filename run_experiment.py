#!/usr/bin/env python3
"""Unified entry point for the experiment roadmap (AMOS22 MRI pancreas).

    python run_experiment.py --exp 1                 # full 5-fold baseline
    python run_experiment.py --exp 1 --folds 0       # smoke test on fold 0 only
    python run_experiment.py --exp 6 --device cpu    # transformer backbones

Paused experiments (not selectable here): 4 (ROI -> duct), 5 (clDice).
"""

import argparse

from experiments import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Run an AMOS MRI pancreas experiment (1/2/3/6)")
    parser.add_argument("--exp", type=int, required=True, choices=list(EXPERIMENTS),
                        help="Experiment number: 1 baseline, 2 ResEnc M/L, 3 spacing, 6 transformers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' (default, auto-detected) or 'cpu'")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="folds to train (default: all 5, the official CV split)")
    args = parser.parse_args()

    EXPERIMENTS[args.exp].run(device=args.device, folds=args.folds)


if __name__ == "__main__":
    main()
