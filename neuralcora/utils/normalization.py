"""Normalization helpers for NeuralCora datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class NormalizationStats:
    mean: float
    std: float
    eps: float = 1e-6

    def apply(self, array: torch.Tensor) -> torch.Tensor:
        return (array - self.mean) / max(self.std, self.eps)

    def invert(self, array: torch.Tensor) -> torch.Tensor:
        return array * max(self.std, self.eps) + self.mean


def _safe_nan_stats(array: np.ndarray) -> tuple[float, float]:
    mean = float(np.nanmean(array))
    std = float(np.nanstd(array))
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def compute_global_stats(data: np.ndarray, mask: Optional[np.ndarray] = None) -> NormalizationStats:
    """Compute global mean/std while being robust to degenerate masks."""

    if mask is not None:
        mask_bool = mask.astype(bool)
        if mask_bool.any():
            masked = np.where(mask_bool, data, np.nan)
            mean, std = _safe_nan_stats(masked)
        else:  # fall back to unmasked data if mask removes everything
            mean, std = _safe_nan_stats(data)
    else:
        mean, std = _safe_nan_stats(data)
    return NormalizationStats(mean=mean, std=std)


def apply_global_normalization(
    tensor: torch.Tensor, stats: NormalizationStats
) -> torch.Tensor:
    """Apply normalization returning a new tensor."""
    return stats.apply(tensor)
