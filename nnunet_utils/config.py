"""nnU-Net v2 paths and dataset constants.

nnU-Net reads ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results`` from
environment variables **at import time**. :func:`setup_paths` must therefore be
called before importing any ``nnunetv2`` module (the entry points do this first).
"""

import os
from pathlib import Path

#: dataset identifier used by every experiment (nnU-Net convention Dataset###_Name)
DATASET_NAME = "Dataset150_PancreasDuct"
DATASET_ID = 150

#: segmentation targets are the thin duct structures, not the whole pancreas
LABELS = {"background": 0, "pancreatic_duct": 1, "bile_duct": 2}
NUM_CLASSES = len(LABELS)

#: local base folders (kept inside the project so everything is self-contained)
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "nnUNet_raw"
PREPROCESSED_DIR = PROJECT_DIR / "nnUNet_preprocessed"
RESULTS_DIR = PROJECT_DIR / "nnUNet_results"

#: default plans identifier produced by ``nnUNetv2_plan_and_preprocess``
DEFAULT_PLANS = "nnUNetPlans"


def setup_paths(raw_dir=None, preprocessed_dir=None, results_dir=None):
    """Point nnU-Net at the local raw / preprocessed / results folders.

    The values are exported as environment variables, so they are inherited by
    the ``nnUNetv2_*`` subprocesses launched by the experiment scripts.
    """
    os.environ["nnUNet_raw"] = str(raw_dir or RAW_DIR)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir or PREPROCESSED_DIR)
    os.environ["nnUNet_results"] = str(results_dir or RESULTS_DIR)
