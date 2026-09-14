#!/usr/bin/env python3
"""Unified entry point that runs one experiment from the technical roadmap.

    python run_experiment.py --exp 1 --data_dir data --device cuda
    python run_experiment.py --exp 6 --data_dir data --device cuda --epochs 200
"""

import argparse

from experiments import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Run a pancreas-segmentation experiment (1-6)")
    parser.add_argument("--exp", type=int, required=True, choices=list(EXPERIMENTS),
                        help="Experiment number (1-6)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the number of training epochs")
    args = parser.parse_args()

    module = EXPERIMENTS[args.exp]

    kwargs = {"data_dir": args.data_dir, "device": args.device}
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.epochs:
        kwargs["epochs"] = args.epochs

    module.run(**kwargs)


if __name__ == "__main__":
    main()
