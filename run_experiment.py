#!/usr/bin/env python3
"""Unified entry point for the six-experiment roadmap.

    python run_experiment.py --exp 1
    python run_experiment.py --exp 6 --device cuda --epochs 500
"""

import argparse

from experiments import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Run a pancreas-segmentation experiment (1-6)")
    parser.add_argument("--exp", type=int, required=True, choices=list(EXPERIMENTS),
                        help="Experiment number (1-6)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' (default, auto-detected) or 'cpu'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training length (edits the plans file)")
    args = parser.parse_args()

    EXPERIMENTS[args.exp].run(device=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
