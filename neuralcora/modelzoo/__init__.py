"""Model registry for NeuralCora."""
from __future__ import annotations

from typing import Dict, Type

from .simple_cnn import SimpleCNNForecaster

_MODEL_REGISTRY: Dict[str, Type[SimpleCNNForecaster]] = {
    "simple_cnn": SimpleCNNForecaster,
}


def register_model(name: str, model_cls: Type[SimpleCNNForecaster]) -> None:
    key = name.lower()
    if key in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' already registered")
    _MODEL_REGISTRY[key] = model_cls


def get_model_class(name: str):
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' is not registered. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[key]


__all__ = ["SimpleCNNForecaster", "register_model", "get_model_class"]
