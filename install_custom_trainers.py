#!/usr/bin/env python3
"""Install the custom trainers into the nnU-Net v2 package.

``nnUNetv2_train`` and ``nnUNetv2_predict`` resolve a trainer by name *inside*
the installed ``nnunetv2`` package (``nnunetv2/training/nnUNetTrainer/...``).
Custom trainers therefore must be copied there once. This is the official
extension mechanism described in the nnU-Net v2 documentation.

Usage:
    python install_custom_trainers.py
"""

import shutil
from pathlib import Path

import nnunetv2


def install():
    """Copy custom trainer modules into the nnunetv2 package (idempotent)."""
    here = Path(__file__).resolve().parent
    src_dir = here / "custom_trainers"
    dst_dir = (Path(nnunetv2.__file__).resolve().parent
               / "training" / "nnUNetTrainer" / "variants")

    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(src_dir.glob("nnUNetTrainer_*.py")):
        dst = dst_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)

    return copied


def main():
    copied = install()
    for dst in copied:
        print(f"installed {dst.name} -> {dst}")
    print("\nCustom trainers are now resolvable by name, e.g.:")
    print("  nnUNetv2_train 150 3d_fullres 0 -tr nnUNetTrainer_DiceCEclDice")
    print("  nnUNetv2_train 150 3d_fullres 0 -tr nnUNetTrainer_UNETR")


if __name__ == "__main__":
    main()
