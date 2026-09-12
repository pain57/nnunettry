# nnU-Net for 3D Pancreas Segmentation

A PyTorch implementation of nnU-Net (no-new-Net) for 3D pancreas segmentation in CT images.

## Overview

nnU-Net is a self-configuring deep learning framework for biomedical image segmentation. This implementation includes:

- **3D U-Net** with encoder-decoder architecture and skip connections
- **Deep supervision** with auxiliary segmentation heads at intermediate decoder stages
- **Instance normalization + LeakyReLU** as in the original nnU-Net
- **Soft Dice + Cross-Entropy** composite loss
- **Polynomial LR decay** with Nesterov SGD
- **3D data augmentation**: flip, rotation, scaling, elastic deformation, gamma, noise, blur
- **Sliding window inference** with Gaussian importance weighting
- **Connected-component post-processing** for false positive removal

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data

```bash
python generate_synthetic_data.py --num_train 20 --num_test 4
```

This creates a synthetic pancreas CT dataset in `data/` following the nnU-Net convention:
```
data/
├── imagesTr/       # Training images
├── labelsTr/       # Training labels
├── imagesTs/       # Test images
└── dataset.json    # Metadata
```

### 2. Train

```bash
# Quick test (CPU, few epochs):
python train.py --data_dir data --output_dir runs/exp1 --epochs 10 --batch_size 1 --device cpu

# Full training (GPU recommended):
python train.py --data_dir data --output_dir runs/exp1 --epochs 1000 --batch_size 2 --device cuda
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Dataset directory with `dataset.json` |
| `--output_dir` | `runs/exp1` | Checkpoint output directory |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `2` | Batch size (reduce for smaller GPU) |
| `--lr` | `0.01` | Initial learning rate |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--val_split` | `0.2` | Validation split fraction |
| `--resume` | `None` | Resume from checkpoint path |

### 3. Predict

```bash
# Single case:
python predict.py --input data/imagesTs/pancreas_test_000_0000.nii.gz \
                  --output predictions/ \
                  --checkpoint runs/exp1/checkpoint_best.pth

# Entire directory:
python predict.py --input data/imagesTs/ \
                  --output predictions/ \
                  --checkpoint runs/exp1/checkpoint_best.pth \
                  --device cuda
```

## Using Real Data

To use your own pancreas CT dataset:

1. Organize data in nnU-Net format:
   ```
   your_data/
   ├── imagesTr/      # Training images (*_0000.nii.gz)
   ├── labelsTr/      # Training labels (*.nii.gz)
   ├── imagesTs/      # Test images (optional)
   └── dataset.json
   ```

2. Edit `dataset.json`:
   ```json
   {
     "name": "YourPancreasDataset",
     "channel_names": {"0": "CT"},
     "labels": {"background": 0, "pancreas": 1},
     "numTraining": 100,
     "training": [
       {"image": "./imagesTr/case_001_0000.nii.gz",
        "label": "./labelsTr/case_001.nii.gz"}
     ],
     "test": ["./imagesTs/case_test_001_0000.nii.gz"]
   }
   ```

3. Train: `python train.py --data_dir your_data --output_dir runs/your_exp`

## Project Structure

```
nnunet/
├── config.py                  # Configuration dataclass
├── dataset/
│   ├── preprocessing.py       # CT normalization, resampling, cropping
│   ├── augmentation.py        # 3D augmentation transforms
│   └── patch_sampler.py       # Patch-based data loading
├── network/
│   ├── blocks.py              # ConvBlock, UpsampleBlock, SegmentationHead
│   ├── unet3d.py              # 3D U-Net with deep supervision
│   └── initialization.py      # Kaiming weight init
├── training/
│   ├── trainer.py             # Training loop + validation
│   ├── losses.py              # Dice, CE, DeepSupervision losses
│   └── lr_scheduler.py        # Polynomial LR decay
├── inference/
│   └── predictor.py           # Sliding window + post-processing
└── utils/
    ├── metrics.py              # Dice score, Hausdorff distance
    └── nifti_io.py            # NIfTI I/O via nibabel
```

## Architecture

The 3D U-Net follows the nnU-Net design:

- **Encoder**: 5 stages, each with 2× Conv3d → InstanceNorm → LeakyReLU, stride-2 downsampling from stage 2
- **Bottleneck**: 2× ConvBlock at 16× downsampled resolution
- **Decoder**: 5 stages with trilinear upsampling + skip connections
- **Deep Supervision**: Auxiliary heads at the 2 lowest-resolution decoder outputs
- **Initial features**: 32 (doubles each stage → 32, 64, 128, 256, 320 at bottleneck)

## References

- Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods* 18, 203–211 (2021).
- Original nnU-Net: https://github.com/MIC-DKFZ/nnUNet
