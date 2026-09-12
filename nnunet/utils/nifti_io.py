import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_nifti(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data_array, affine_matrix)."""
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, filepath: str, header: Optional[nib.Nifti1Header] = None):
    """Save a numpy array as a NIfTI file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    nib.save(img, filepath)


def save_segmentation_nifti(segmentation: np.ndarray, affine: np.ndarray, filepath: str):
    """Save a segmentation mask as NIfTI (integer labels)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # If argmax output is one-hot, squeeze channel dim
    if segmentation.ndim == 4 and segmentation.shape[-1] > 1:
        segmentation = np.argmax(segmentation, axis=-1)
    img = nib.Nifti1Image(segmentation.astype(np.uint8), affine)
    nib.save(img, filepath)


def get_nifti_info(filepath: str) -> dict:
    """Get metadata about a NIfTI file."""
    img = nib.load(filepath)
    return {
        "shape": img.shape,
        "spacing": tuple(img.header.get_zooms()[:3]),
        "affine": img.affine,
        "header": img.header,
    }
