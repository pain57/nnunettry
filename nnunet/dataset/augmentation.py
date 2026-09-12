"""
3D image augmentation transforms for CT pancreas segmentation.

All transforms operate on PyTorch tensors of shape (C, D, H, W)
and support paired image+label transformation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


class ComposeTransforms:
    """Compose multiple augmentation transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomFlip:
    """Randomly flip along each spatial axis with probability p."""

    def __init__(self, p: float = 0.5, axes: Tuple[int, ...] = (0, 1, 2)):
        self.p = p
        self.axes = axes

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # axes correspond to D, H, W dimensions (index 1, 2, 3 in (C, D, H, W))
        spatial_axes = [a + 1 for a in self.axes]  # shift by channel dim
        for ax in spatial_axes:
            if torch.rand(1).item() < self.p:
                image = torch.flip(image, dims=[ax])
                label = torch.flip(label, dims=[ax])
        return image, label


class RandomRotation3D:
    """Random rotation in the axial plane (H, W axes)."""

    def __init__(self, max_angle_deg: float = 15.0, p: float = 0.5):
        self.max_angle = np.deg2rad(max_angle_deg)
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label

        angle = (torch.rand(1).item() * 2 - 1) * self.max_angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ], dtype=torch.float32)

        # Build affine grid — 3D requires Nx3x4 (rotation [3x3] + translation [3])
        theta = torch.zeros(1, 3, 4, dtype=torch.float32)
        theta[:, :3, :3] = rot_matrix
        for i in range(2):  # for image and label
            data = image if i == 0 else label
            # data shape: (C, D, H, W); affine_grid needs (N, C, D, H, W) = 5 values
            grid = F.affine_grid(theta, [1, data.shape[0]] + list(data.shape[1:]),
                                 align_corners=False)
            transformed = F.grid_sample(
                data.unsqueeze(0) if data.dim() == 4 else data,
                grid, mode="bilinear" if i == 0 else "nearest",
                align_corners=False, padding_mode="zeros"
            )
            if i == 0:
                image = transformed.squeeze(0) if transformed.shape[0] == 1 else transformed
            else:
                label = transformed.squeeze(0) if transformed.shape[0] == 1 else transformed

        return image, label


class RandomScaling3D:
    """Random scaling in each spatial dimension."""

    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.15), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label

        scale = 1.0 + (torch.rand(3) * (self.scale_range[1] - self.scale_range[0])
                        + self.scale_range[0] - 1.0)

        theta = torch.zeros(1, 3, 4, dtype=torch.float32)
        theta[:, 0, 0] = scale[0]
        theta[:, 1, 1] = scale[1]
        theta[:, 2, 2] = scale[2]
        for i in range(2):
            data = image if i == 0 else label
            spatial_size = data.shape  # (C, D, H, W)
            grid = F.affine_grid(theta, [1, data.shape[0]] + list(data.shape[1:]),
                                 align_corners=False)
            transformed = F.grid_sample(
                data.unsqueeze(0),
                grid, mode="bilinear" if i == 0 else "nearest",
                align_corners=False, padding_mode="zeros"
            )
            if i == 0:
                image = transformed.squeeze(0)
            else:
                label = transformed.squeeze(0)

        return image, label


class RandomGamma:
    """Apply random gamma correction to image only."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label
        gamma = np.random.uniform(*self.gamma_range)
        # Ensure positive values
        img_min = image.min()
        image_shifted = image - img_min + 1e-6
        image = image_shifted.pow(gamma) + img_min - 1e-6
        return image, label


class RandomGaussianNoise:
    """Add Gaussian noise to image."""

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.1), p: float = 0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label
        std = np.random.uniform(*self.std_range)
        if std > 0:
            noise = torch.randn_like(image) * std
            image = image + noise
        return image, label


class RandomGaussianBlur:
    """Apply Gaussian blur to image using scipy (applied per slice for 3D)."""

    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.2):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label
        sigma = np.random.uniform(*self.sigma_range)
        img_np = image.numpy()
        for c in range(img_np.shape[0]):
            img_np[c] = gaussian_filter(img_np[c], sigma=sigma)
        image = torch.from_numpy(img_np)
        return image, label


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        p: float = 0.3,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        mean = image.mean()
        image = (image - mean) * contrast + mean * brightness
        return image, label


class ElasticDeformation3D:
    """
    3D elastic deformation using B-spline displacement fields.
    Implemented via scipy map_coordinates for smooth deformation.
    """

    def __init__(
        self,
        num_control_points: int = 4,
        deformation_scale: float = 10.0,
        p: float = 0.3,
    ):
        self.num_control_points = num_control_points
        self.deformation_scale = deformation_scale
        self.p = p

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return image, label

        shape_dhw = image.shape[1:]  # (D, H, W) — spatial dims
        shape_arr = np.array(shape_dhw)

        # Create coarse displacement field and upsample
        control_spacing = np.maximum(shape_arr // self.num_control_points, 2)
        control_shape = tuple(np.maximum(shape_arr // control_spacing, 2))

        # Use fewer control points for smaller volumes
        control_shape = tuple(min(cs, 6) for cs in control_shape)

        displacements = []
        for _ in range(3):
            dx = np.random.randn(*control_shape) * self.deformation_scale
            # Upsample to full resolution
            zoom_factors = tuple(s / c for s, c in zip(shape_dhw, control_shape))
            dx_full = zoom(dx, zoom_factors, order=3)
            displacements.append(dx_full)

        # Apply deformation
        grid = np.meshgrid(
            np.arange(shape_dhw[0]), np.arange(shape_dhw[1]), np.arange(shape_dhw[2]),
            indexing="ij"
        )
        indices = [g + d for g, d in zip(grid, displacements)]

        for i in range(2):
            data = image if i == 0 else label
            data_np = data.numpy()
            # Handle channel dimension: apply deformation per channel
            if data_np.ndim == 4:
                deformed = np.zeros_like(data_np)
                for c in range(data_np.shape[0]):
                    deformed[c] = map_coordinates(
                        data_np[c], indices, order=1 if i == 0 else 0, mode="nearest"
                    )
            else:
                deformed = map_coordinates(
                    data_np, indices, order=1 if i == 0 else 0, mode="nearest"
                )
            if i == 0:
                image = torch.from_numpy(deformed.astype(np.float32))
            else:
                label = torch.from_numpy(deformed.astype(np.float32))

        return image, label


def get_training_augmentation(p: float = 0.5) -> ComposeTransforms:
    """Build the default training augmentation pipeline."""
    return ComposeTransforms([
        RandomFlip(p=0.5, axes=(0, 1, 2)),
        RandomRotation3D(max_angle_deg=15.0, p=p),
        RandomScaling3D(scale_range=(0.85, 1.15), p=p),
        ElasticDeformation3D(num_control_points=4, deformation_scale=10.0, p=0.2),
        RandomGamma(gamma_range=(0.7, 1.5), p=0.3),
        RandomGaussianNoise(std_range=(0.0, 0.1), p=0.3),
        RandomGaussianBlur(sigma_range=(0.5, 1.0), p=0.2),
        RandomBrightnessContrast(brightness_range=(0.85, 1.15), contrast_range=(0.85, 1.15), p=0.15),
    ])
