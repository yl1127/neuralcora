"""Dataset utilities for loading CORA coastal zeta grids."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neuralcora.utils.normalization import NormalizationStats, compute_global_stats

__all__ = ["CoraDataset", "CoraDatasetMetadata"]


@dataclass
class CoraDatasetMetadata:
    time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    mask: Optional[np.ndarray]


class CoraDataset(Dataset):
    """Sequence-to-sequence dataset for CORA NetCDF files."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        split: str,
        input_steps: int,
        forecast_steps: int,
        stride: int = 1,
        normalization: str = "global",
        stats: Optional[NormalizationStats] = None,
        mask_path: Optional[str | Path] = None,
        split_fractions: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"CORA dataset not found at {self.data_path}")
        self.mask_path = Path(mask_path) if mask_path else None
        self.split = split
        self.input_steps = int(input_steps)
        self.forecast_steps = int(forecast_steps)
        self.stride = int(stride)
        self.normalization = normalization

        self._raw_array, metadata = self._load_arrays()
        self.metadata = metadata

        if self.input_steps < 1 or self.forecast_steps < 1:
            raise ValueError("input_steps and forecast_steps must be >= 1")
        if self._raw_array.shape[0] < self.input_steps + self.forecast_steps:
            raise ValueError(
                "Insufficient timesteps for requested input/forecast window."
            )

        if normalization == "global":
            if stats is None:
                stats = compute_global_stats(self._raw_array, metadata.mask)
            self.stats = stats
            normalized = (self._raw_array - stats.mean) / max(stats.std, stats.eps)
        else:
            self.stats = stats
            normalized = self._raw_array

        normalized = np.nan_to_num(normalized, nan=0.0, copy=False)
        self._tensor = torch.from_numpy(normalized.astype(np.float32, copy=False))

        self._indices = self._build_indices(split, split_fractions)

    def _load_arrays(self) -> Tuple[np.ndarray, CoraDatasetMetadata]:
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Loading CORA NetCDF files requires xarray. Install with 'pip install xarray netcdf4'."
            ) from exc

        with xr.open_dataset(self.data_path) as dataset:
            if "zeta" not in dataset:
                raise KeyError("Expected variable 'zeta' in CORA dataset")
            zeta = dataset["zeta"].astype("float32").values
            time = dataset["time"].values if "time" in dataset else None
            latitude = dataset["latitude"].values if "latitude" in dataset else None
            longitude = dataset["longitude"].values if "longitude" in dataset else None

        mask_array = None
        if self.mask_path is not None and self.mask_path.exists():
            with xr.open_dataset(self.mask_path) as mask_ds:
                mask_var_name = next(iter(mask_ds.data_vars))
                mask_array = mask_ds[mask_var_name].astype("float32").values
                mask_array = mask_array.astype(bool)

        metadata = CoraDatasetMetadata(
            time=time,
            latitude=latitude,
            longitude=longitude,
            mask=mask_array,
        )
        return zeta, metadata

    def _build_indices(
        self, split: str, split_fractions: Optional[Mapping[str, float]]
    ) -> List[int]:
        split = split.lower()
        fractions = split_fractions or {"train": 0.7, "val": 0.15, "test": 0.15}
        total = sum(fractions.values())
        if total <= 0:
            raise ValueError("split fractions must be positive")
        norm_fractions: Dict[str, float] = {
            key.lower(): value / total for key, value in fractions.items()
        }
        samples = self._raw_array.shape[0] - (self.input_steps + self.forecast_steps) + 1
        if samples <= 0:
            raise ValueError("Not enough samples to build windows with current settings")
        cumulative: Dict[str, Tuple[float, float]] = {}
        start = 0.0
        for name, fraction in norm_fractions.items():
            end = start + fraction
            cumulative[name] = (start, end)
            start = end
        if split not in cumulative:
            raise KeyError(
                f"Split '{split}' not found in split fractions {list(cumulative.keys())}"
            )
        frac_start, frac_end = cumulative[split]
        start_idx = int(np.floor(frac_start * samples))
        end_idx = int(np.floor(frac_end * samples)) if split != list(cumulative.keys())[-1] else samples
        indices: Iterable[int] = range(start_idx, max(end_idx, start_idx + 1), self.stride)
        return list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base_idx = self._indices[index]
        input_slice = slice(base_idx, base_idx + self.input_steps)
        target_slice = slice(
            base_idx + self.input_steps,
            base_idx + self.input_steps + self.forecast_steps,
        )
        inputs = self._tensor[input_slice]
        targets = self._tensor[target_slice]
        # Convert to (channels, H, W)
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        return {
            "inputs": inputs,
            "targets": targets,
            "input_start_index": base_idx,
            "target_start_index": base_idx + self.input_steps,
        }

    @property
    def normalization_stats(self) -> Optional[NormalizationStats]:
        return self.stats
