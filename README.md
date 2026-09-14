# nnU-Net 3D Pancreas Segmentation — Experiment Framework

A self-contained PyTorch implementation of nnU-Net for 3D pancreas segmentation
in CT, reorganized around a six-step technical roadmap. Each step maps to one
experiment module and to one knob in [`nnunet/config.py`](nnunet/config.py).

## Technical roadmap

| # | Experiment | What it changes | Where |
|---|------------|-----------------|-------|
| 1 | **Baseline** — nnU-Net v2, 3D fullres, PlainConvUNet | default backbone + Dice/CE | [exp1_baseline.py](experiments/exp1_baseline.py) |
| 2 | **Backbone** — ResEnc M / L | residual-encoder U-Net (M/L sizes) | [exp2_backbone.py](experiments/exp2_backbone.py) |
| 3 | **Target spacing** | resample to isotropic / anisotropic spacing | [exp3_spacing.py](experiments/exp3_spacing.py) |
| 4 | **ROI / coarse-to-fine** | two-stage localization + ROI sampling | [exp4_roi.py](experiments/exp4_roi.py) |
| 5 | **Continuity** — Dice+CE vs +clDice | add differentiable clDice term | [exp5_cldice.py](experiments/exp5_cldice.py) |
| 6 | **Other models** — UNETR / SwinUNETR / MedNeXt | alternative backbones | [exp6_models.py](experiments/exp6_models.py) |

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Generate synthetic data (optionally add a thin duct for Exp 3/5)
python generate_synthetic_data.py --num_train 20 --num_test 4 --add_duct

# 2. Run an experiment (unified entry point)
python run_experiment.py --exp 1 --data_dir data --device cuda
python run_experiment.py --exp 6 --data_dir data --device cuda --epochs 200

# or train / predict directly
python train.py --data_dir data --output_dir runs/exp1 --backbone plain --epochs 1000
python predict.py --input data/imagesTs --output predictions \
    --checkpoint runs/exp1/checkpoint_best.pth

# 3. Evaluate
python evaluate.py --pred_dir predictions --gt_dir data/labelsTs
```

## Experiment notes

- **Exp 1** — plain baseline at native spacing; the reference every other
  experiment is compared against.
- **Exp 2** — `resenc_m` vs `resenc_l` differ only in encoder channel depth and
  block count (see `nnunet/network/resenc.py`).
- **Exp 3** — `--target_spacing` resamples volumes before cropping. Finer
  isotropic spacing preserves thin structures at higher memory cost.
- **Exp 4** — a coarse model localizes the pancreas, a fine model (trained with
  `--use_roi`, sampling patches inside the ground-truth ROI) segments the crop.
  Combine them at inference with `predict.py --coarse_checkpoint ...`.
- **Exp 5** — `--loss dice_ce_cldice` adds a differentiable clDice term that
  rewards topological continuity of thin / tubular structures.
- **Exp 6** — transformer backbones (`unetr`, `swin_unetr`) and ConvNeXt
  (`mednext_s`, `mednext_m`). UNETR defaults are reduced to stay tractable on a
  single GPU.

## Project structure

```
胰腺分割/
├── nnunet/
│   ├── config.py                  # single dataclass driving all experiments
│   ├── network/
│   │   ├── unet3d.py              # PlainConvUNet (baseline)
│   │   ├── resenc.py              # ResEncUNet M/L (residual encoder)
│   │   ├── unetr.py               # UNETR (ViT encoder)
│   │   ├── swin_unetr.py          # SwinUNETR (Swin Transformer encoder)
│   │   ├── mednext.py             # MedNeXt S/M (ConvNeXt encoder-decoder)
│   │   ├── blocks.py              # shared conv / residual / upsampling blocks
│   │   └── __init__.py            # build_network factory + registry
│   ├── dataset/
│   │   ├── data_loading.py        # dataset.json loading + target-spacing resampling
│   │   ├── preprocessing.py       # normalization, resampling, cropping
│   │   ├── patch_sampler.py       # patch sampling + ROI focusing
│   │   └── augmentation.py        # 3D augmentation transforms
│   ├── training/
│   │   ├── trainer.py             # training loop + validation
│   │   ├── losses.py              # Dice, CE, clDice, deep supervision
│   │   └── lr_scheduler.py        # polynomial LR decay
│   ├── inference/
│   │   ├── predictor.py           # sliding-window inference
│   │   └── coarse_to_fine.py      # ROI / two-stage inference
│   └── utils/
│       ├── metrics.py             # Dice, HD95, clDice
│       └── nifti_io.py            # NIfTI I/O
├── experiments/                   # one module per roadmap step
│   ├── common.py
│   ├── exp1_baseline.py ... exp6_models.py
├── generate_synthetic_data.py
├── train.py / predict.py / evaluate.py
└── run_experiment.py              # python run_experiment.py --exp N
```

## Using real data

Organize data in the nnU-Net convention (`imagesTr`, `labelsTr`, `imagesTs`,
`dataset.json`) and point `--data_dir` at it. `dataset.json` must contain
`channel_names`, `labels`, `training`, `test` (and optionally `validation`).

## References

- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation." *Nature Methods* 18, 203–211 (2021).
- Shit et al., "clDice — a novel topology-preserving loss function for tubular
  structure segmentation." *CVPR* (2021).
- Hatamizadeh et al., "UNETR: Transformers for 3D medical image segmentation."
  *WACV* (2022); "Swin UNETR: Swin Transformers for semantic segmentation of
  brain tumors in MRI images." *MICCAI BrainLes* (2022).
- Roy et al., "MedNeXt: Transformer-driven scaling of ConvNets for medical image
  segmentation." *MICCAI* (2023).
