"""DataModule-like helper to create DataLoaders for NeuralCora."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from neuralcora.dataset import CoraDataset, get_dataset_class
from neuralcora.utils import DataConfig, NormalizationStats


class CoraDataModule:
    """Light-weight data module to produce train/val/test loaders."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._train: Optional[CoraDataset] = None
        self._val: Optional[CoraDataset] = None
        self._test: Optional[CoraDataset] = None
        self._stats: Optional[NormalizationStats] = None

    @property
    def stats(self) -> Optional[NormalizationStats]:
        return self._stats

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_cls = get_dataset_class(self.config.dataset_name)
        if self._train is None or stage in {"fit", None}:
            self._train = dataset_cls(
                data_path=self.config.dataset_path,
                mask_path=self.config.mask_path,
                split="train",
                input_steps=self.config.input_steps,
                forecast_steps=self.config.forecast_steps,
                stride=self.config.stride,
                normalization=self.config.normalization,
                split_fractions=self.config.split_fractions,
            )
            self._stats = self._train.normalization_stats

        stats = self._stats
        if stage in {"fit", "validate", None}:
            self._val = dataset_cls(
                data_path=self.config.dataset_path,
                mask_path=self.config.mask_path,
                split="val",
                input_steps=self.config.input_steps,
                forecast_steps=self.config.forecast_steps,
                stride=self.config.stride,
                normalization=self.config.normalization,
                stats=stats,
                split_fractions=self.config.split_fractions,
            )
        if stage in {"test", "predict", None}:
            self._test = dataset_cls(
                data_path=self.config.dataset_path,
                mask_path=self.config.mask_path,
                split="test",
                input_steps=self.config.input_steps,
                forecast_steps=self.config.forecast_steps,
                stride=self.config.stride,
                normalization=self.config.normalization,
                stats=stats,
                split_fractions=self.config.split_fractions,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train is None:
            self.setup(stage="fit")
        return DataLoader(
            self._train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val is None:
            self.setup(stage="validate")
        return DataLoader(
            self._val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test is None:
            self.setup(stage="test")
        return DataLoader(
            self._test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def save_stats(self, path: str | Path) -> None:
        if self._stats is None:
            raise RuntimeError("No normalization statistics computed yet.")
        tensor = torch.tensor([self._stats.mean, self._stats.std, self._stats.eps])
        torch.save(tensor, Path(path))

    def load_stats(self, path: str | Path) -> None:
        tensor = torch.load(Path(path))
        self._stats = NormalizationStats(
            mean=float(tensor[0]), std=float(tensor[1]), eps=float(tensor[2])
        )

    def get_dataset(self, split: str) -> CoraDataset:
        split = split.lower()
        if split == "train":
            if self._train is None:
                self.setup(stage="fit")
            if self._train is None:
                raise RuntimeError("Training dataset is unavailable after setup")
            return self._train
        if split == "val":
            if self._val is None:
                self.setup(stage="validate")
            if self._val is None:
                raise RuntimeError("Validation dataset is unavailable after setup")
            return self._val
        if split == "test":
            if self._test is None:
                self.setup(stage="test")
            if self._test is None:
                raise RuntimeError("Test dataset is unavailable after setup")
            return self._test
        raise ValueError(f"Unknown split '{split}'. Expected train, val, or test.")
