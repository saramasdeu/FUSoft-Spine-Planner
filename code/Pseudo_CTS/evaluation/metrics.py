"""
evaluation/metrics.py
---------------------
Metrics for pseudo-CT quality assessment.

All functions receive numpy arrays already co-registered and in the same
voxel space (same shape).  Arrays should be in HU (not normalised).

Metrics
-------
  MAE_global      : mean absolute error over all body voxels (CT > -500 HU)
  MAE_bone        : MAE restricted to bone region (CT > bone_threshold HU)
  MAE_soft_tissue : MAE restricted to soft tissue (-100 to 400 HU)
  SSIM            : structural similarity index
  PSNR            : peak signal-to-noise ratio (dB)
  DSC_bone        : Dice similarity coefficient of binarised bone masks

Usage
-----
    from evaluation.metrics import compute_all
    results = compute_all(pseudo_ct_array, real_ct_array)
    # → dict with keys MAE_global, MAE_bone, MAE_soft_tissue, SSIM, PSNR, DSC_bone
"""

import numpy as np
from skimage.metrics import structural_similarity as _sk_ssim

# HU range used for SSIM / PSNR normalisation (matches training convention)
_HU_MIN: float = -1000.0
_HU_MAX: float =  2000.0
_HU_RANGE: float = _HU_MAX - _HU_MIN


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def _body_mask(ct: np.ndarray) -> np.ndarray:
    """Exclude air voxels (< -500 HU)."""
    return ct > -500.0


def _bone_mask(ct: np.ndarray, threshold: float = 400.0) -> np.ndarray:
    return ct > threshold


def _soft_tissue_mask(ct: np.ndarray, lo: float = -100.0, hi: float = 400.0) -> np.ndarray:
    return (ct >= lo) & (ct <= hi)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None,
) -> float:
    """Mean Absolute Error in HU.  NaN if mask is empty."""
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        return float(np.abs(pred[mask] - target[mask]).mean())
    return float(np.abs(pred - target).mean())


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """3D SSIM over the full volume."""
    return float(
        _sk_ssim(
            target.astype(np.float64),
            pred.astype(np.float64),
            data_range=_HU_RANGE,
        )
    )


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB.  +inf if MSE = 0."""
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(_HU_RANGE / np.sqrt(mse)))


def compute_dsc_bone(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 400.0,
) -> float:
    """Dice Similarity Coefficient of threshold-based bone masks.

    Both pseudo-CT and real CT are binarised at `threshold` HU.
    NaN if neither volume contains bone.
    """
    pred_bone = _bone_mask(pred, threshold)
    true_bone = _bone_mask(target, threshold)
    intersection = int((pred_bone & true_bone).sum())
    denom = int(pred_bone.sum()) + int(true_bone.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * intersection / denom)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_all(
    pred: np.ndarray,
    target: np.ndarray,
    bone_threshold: float = 400.0,
) -> dict:
    """Compute all metrics.

    Parameters
    ----------
    pred           : pseudo-CT array in HU
    target         : real CT array in HU, co-registered with pred
    bone_threshold : HU threshold for bone binary mask (default 400 HU)

    Returns
    -------
    dict with keys:
        MAE_global, MAE_bone, MAE_soft_tissue, SSIM, PSNR, DSC_bone
    """
    pred   = pred.astype(np.float32)
    target = target.astype(np.float32)

    body = _body_mask(target)
    bone = _bone_mask(target, bone_threshold)
    soft = _soft_tissue_mask(target)

    return {
        "MAE_global":      compute_mae(pred, target, body),
        "MAE_bone":        compute_mae(pred, target, bone),
        "MAE_soft_tissue": compute_mae(pred, target, soft),
        "SSIM":            compute_ssim(pred, target),
        "PSNR":            compute_psnr(pred, target),
        "DSC_bone":        compute_dsc_bone(pred, target, bone_threshold),
    }