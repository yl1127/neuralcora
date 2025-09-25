"""Dataset registry for NeuralCora."""
from __future__ import annotations

from typing import Dict, Type

from .cora_dataset import CoraDataset

_DATASET_REGISTRY: Dict[str, Type[CoraDataset]] = {
    "cora": CoraDataset,
}


def register_dataset(name: str, dataset_cls: Type[CoraDataset]) -> None:
    key = name.lower()
    if key in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' already registered")
    _DATASET_REGISTRY[key] = dataset_cls


def get_dataset_class(name: str) -> Type[CoraDataset]:
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered. Available: {list(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[key]


__all__ = ["CoraDataset", "register_dataset", "get_dataset_class"]
