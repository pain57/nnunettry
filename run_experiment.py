#!/usr/bin/env python3
"""Unified entry point for the six-experiment roadmap.

    python run_experiment.py --exp 1
    python run_experiment.py --exp 6 --device cpu --folds 0 1     # subset of folds
"""

import argparse

from experiments import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Run a duct-segmentation experiment (1-6)")
    parser.add_argument("--exp", type=int, required=True, choices=list(EXPERIMENTS),
                        help="Experiment number (1-6)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' (default, auto-detected) or 'cpu'")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="folds to train (default: all 5, the official CV split)")
    args = parser.parse_args()

    EXPERIMENTS[args.exp].run(device=args.device, folds=args.folds)


if __name__ == "__main__":
    main()
