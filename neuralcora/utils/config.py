"""Configuration dataclasses and loader for NeuralCora experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover - fallback handled at runtime
    yaml = None


def _filter_kwargs(dataclass_type, values: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in values.items() if k in allowed}


def _deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class DataConfig:
    """Configuration controlling dataset assembly and dataloading."""

    dataset_name: str = "cora"
    dataset_path: str = "data/NY_2013_180_360.nc"
    mask_path: Optional[str] = "data/real_land_mask_180_360.nc"
    input_steps: int = 1
    forecast_steps: int = 6
    stride: int = 1
    normalization: str = "global"
    stats_path: Optional[str] = None
    batch_size: int = 4
    num_workers: int = 0
    split_fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )

    def validate(self) -> None:
        total = sum(self.split_fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split_fractions must sum to 1.0, received total={total:.4f}"
            )
        if self.input_steps < 1:
            raise ValueError("input_steps must be >= 1")
        if self.forecast_steps < 1:
            raise ValueError("forecast_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >=1")
        if self.stride < 1:
            raise ValueError("stride must be >=1")


@dataclass
class ModelConfig:
    """Configuration for model instantiation."""

    name: str = "simple_cnn"
    input_channels: int = 1
    output_channels: int = 6
    hidden_channels: Sequence[int] = field(default_factory=lambda: (32, 64, 64))
    kernel_size: int = 3
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for trainer and optimization hyperparameters."""

    trainer_name: str = "basic"
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    log_interval: int = 10
    save_dir: str = "runs"
    gradient_clip_norm: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Top-level configuration aggregating all sub-configs."""

    seed: int = 42
    run_name: str = "cora-demo"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "ExperimentConfig":
        if raw is None:
            raw = {}
        data_cfg = DataConfig(**_filter_kwargs(DataConfig, raw.get("data", {})))
        model_cfg = ModelConfig(**_filter_kwargs(ModelConfig, raw.get("model", {})))
        training_cfg = TrainingConfig(
            **_filter_kwargs(TrainingConfig, raw.get("training", {}))
        )
        other_kwargs = _filter_kwargs(cls, raw)
        other_kwargs.pop("data", None)
        other_kwargs.pop("model", None)
        other_kwargs.pop("training", None)
        config = cls(
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
            **other_kwargs,
        )
        config.data.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "run_name": self.run_name,
            "notes": self.notes,
            "data": vars(self.data).copy(),
            "model": {
                "name": self.model.name,
                "input_channels": self.model.input_channels,
                "output_channels": self.model.output_channels,
                "hidden_channels": list(self.model.hidden_channels),
                "kernel_size": self.model.kernel_size,
                "dropout": self.model.dropout,
            },
            "training": {
                "trainer_name": self.training.trainer_name,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "device": self.training.device,
                "log_interval": self.training.log_interval,
                "save_dir": self.training.save_dir,
                "gradient_clip_norm": self.training.gradient_clip_norm,
            },
        }


def load_config(
    config: Optional[Path | str | Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ExperimentConfig:
    """Load an experiment configuration from YAML or mapping.

    Parameters
    ----------
    config: Path | str | Mapping | None
        Path to a YAML file or in-memory mapping. Defaults to ``None`` which
        returns the library defaults.
    overrides: Mapping | None
        Optional deep overrides applied on top of the loaded config.
    """

    if isinstance(config, Mapping):
        config_dict: Dict[str, Any] = dict(config)
    elif config is None:
        config_dict = {}
    else:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load YAML config files. Install with 'pip install pyyaml'."
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

    if overrides:
        config_dict = _deep_update(config_dict, dict(overrides))

    return ExperimentConfig.from_dict(config_dict)
