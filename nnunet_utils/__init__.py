"""Thin helpers around the official nnU-Net v2 package.

Nothing here reimplements nnU-Net. The helpers only:
  * point nnU-Net at local raw / preprocessed / results folders,
  * wrap the official ``nnUNetv2_*`` command-line entry points,
  * compute final evaluation metrics (Dice / HD95 / clDice) that nnU-Net
    does not expose as a standalone command.
"""
