"""Shared helpers that drive the official nnU-Net v2 command-line tools.

Nothing here reimplements nnU-Net: planning / preprocessing / training /
prediction are delegated to the installed ``nnUNetv2_*`` entry points. These
helpers only assemble the right command lines and, for the custom trainers,
copy them into the nnunetv2 package once.
"""

import subprocess

from nnunet_utils.config import DATASET_ID, DEFAULT_PLANS, setup_paths

# Point nnU-Net at the local folders before any nnunetv2 import.
setup_paths()

#: official nnU-Net v2 auto-creates a standard 5-fold split; train all folds.
ALL_FOLDS = list(range(5))


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def plan_and_preprocess(dataset_id=DATASET_ID, configurations=("3d_fullres",),
                        plans_identifier=DEFAULT_PLANS, target_spacing=None, planner=None):
    """Run the official planning + preprocessing (``nnUNetv2_plan_and_preprocess``).

    ``planner`` selects the experiment planner (e.g. ``nnUNetPlannerResEncM``);
    when omitted, nnU-Net uses its default planner.
    """
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id),
           "-c"] + list(configurations) + [
           "--verify_dataset_integrity",
           "-overwrite_plans_name", plans_identifier]
    if planner is not None:
        cmd += ["-pl", planner]
    if target_spacing is not None:
        cmd += ["-overwrite_target_spacing"] + [str(s) for s in target_spacing]
    run(cmd)


def train(trainer_name, dataset_id=DATASET_ID, configuration="3d_fullres", folds=None,
          plans_identifier=DEFAULT_PLANS, device=None):
    """Train one trainer on the official 5-fold cross-validation split."""
    if folds is None:
        folds = ALL_FOLDS
    elif isinstance(folds, int):
        folds = [folds]

    for fold in folds:
        cmd = ["nnUNetv2_train", str(dataset_id), configuration, str(fold),
               "-tr", trainer_name, "-p", plans_identifier]
        if device and device != "cuda":  # nnU-Net auto-detects CUDA by default
            cmd += ["-device", device]
        run(cmd)


def predict(input_folder, output_folder, dataset_id=DATASET_ID,
            configuration="3d_fullres", trainer_name="nnUNetTrainer",
            plans_identifier=DEFAULT_PLANS, folds=ALL_FOLDS,
            checkpoint="checkpoint_final.pth"):
    """Run the official ``nnUNetv2_predict`` (ensemble over the given folds)."""
    cmd = ["nnUNetv2_predict", "-i", str(input_folder), "-o", str(output_folder),
           "-d", str(dataset_id), "-c", configuration, "-tr", trainer_name,
           "-p", plans_identifier, "-f"] + [str(f) for f in folds] + ["-chk", checkpoint]
    run(cmd)


def ensure_custom_trainers_installed():
    """Copy the custom trainers into the nnunetv2 package (idempotent)."""
    from install_custom_trainers import install
    install()
