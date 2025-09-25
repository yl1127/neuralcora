"""NeuralCora package public API."""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from .run import evaluate, predict, train

try:
    __version__ = version("neuralcora")
except PackageNotFoundError:  # pragma: no cover - used for local editable installs
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "train",
    "evaluate",
    "predict",
]


def _lazy_import(module: str):
    """Import helper to avoid loading heavy deps during package import."""
    return import_module(module)


# Convenience getters for commonly used components

def get_dataset_class(name: str):
    """Return a dataset class registered under ``neuralcora.dataset``."""
    dataset_module = _lazy_import("neuralcora.dataset")
    return dataset_module.get_dataset_class(name)


def get_model_class(name: str):
    """Return a model class registered under ``neuralcora.modelzoo``."""
    model_module = _lazy_import("neuralcora.modelzoo")
    return model_module.get_model_class(name)


def get_trainer_class(name: str = "basic"):
    training_module = _lazy_import("neuralcora.training")
    return training_module.get_trainer_class(name)
