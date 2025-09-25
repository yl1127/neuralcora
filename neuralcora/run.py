"""High-level run helpers mirroring NeuralHydrology entry points."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from neuralcora.datautils import CoraDataModule
from neuralcora.modelzoo import get_model_class
from neuralcora.training import get_trainer_class
from neuralcora.utils import ExperimentConfig, load_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_components(config: ExperimentConfig):
    data_module = CoraDataModule(config.data)
    model_cls = get_model_class(config.model.name)
    input_channels = config.model.input_channels or config.data.input_steps
    output_channels = config.model.output_channels or config.data.forecast_steps
    model = model_cls(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=config.model.hidden_channels,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
    )
    trainer_cls = get_trainer_class(config.training.trainer_name)
    trainer = trainer_cls(config.training)
    return model, data_module, trainer


def train(
    config: Optional[str | Path | Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the training loop for a given experiment configuration."""

    experiment_config = load_config(config, overrides)
    _set_seed(experiment_config.seed)
    model, data_module, trainer = _build_components(experiment_config)
    history = trainer.fit(model, data_module)
    return {
        "config": experiment_config,
        "history": history,
        "model_state_dict": model.state_dict(),
    }


def evaluate(
    config: Optional[str | Path | Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model on the test split."""

    experiment_config = load_config(config, overrides)
    _set_seed(experiment_config.seed)
    model, data_module, trainer = _build_components(experiment_config)
    if checkpoint_path:
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict)
    if data_module.stats is None:
        data_module.setup(stage="fit")
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    loss, metrics = trainer.evaluate(model, test_loader, stats=data_module.stats)
    return {"loss": loss, "metrics": metrics}


def predict(
    config: Optional[str | Path | Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> torch.Tensor:
    """Generate predictions for the test split."""

    experiment_config = load_config(config, overrides)
    _set_seed(experiment_config.seed)
    model, data_module, trainer = _build_components(experiment_config)
    if checkpoint_path:
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict)
    if data_module.stats is None:
        data_module.setup(stage="fit")
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    return trainer.predict(model, test_loader)
