# NeuralCora Configuration Guide

NeuralCora experiments are controlled through nested configuration blocks that map directly to the `ExperimentConfig`, `DataConfig`, `ModelConfig`, and `TrainingConfig` dataclasses defined in `neuralcora/utils/config.py`. Configurations can be stored as YAML or passed as Python dictionaries.

## Top-level Keys

```yaml
run_name: my-experiment
seed: 42
notes: Optional description

data: { ... }
model: { ... }
training: { ... }
```

- `run_name` *(str, default `"cora-demo"`)*: Folder name appended to `training.save_dir` for checkpoints and artefacts.
- `seed` *(int, default `42`)*: Seed forwarded to Python, NumPy, and PyTorch (CPU/GPU) for reproducibility.
- `notes` *(str or null)*: Optional metadata saved alongside the config.

## Data Block (`DataConfig`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | str | `"cora"` | Registry key resolved via `neuralcora.dataset.get_dataset_class`. |
| `dataset_path` | str | `data/NY_2013_180_360.nc` | NetCDF path holding the CORA zeta cube. |
| `mask_path` | str | `data/real_land_mask_180_360.nc` | Optional land/sea mask aligned with the dataset grid. Set to `null` to disable masking. |
| `input_steps` | int | `1` | Number of past timesteps (channels) provided to the model. Must be `>= 1`. |
| `forecast_steps` | int | `6` | Prediction horizon length (output channels). |
| `stride` | int | `1` | Temporal shift between consecutive training windows. Larger values reduce sample overlap. |
| `normalization` | str | `"global"` | Normalisation strategy. `global` computes mean/std over the full cube; use `none` to skip scaling. |
| `stats_path` | str or null | `null` | Optional torch file storing `[mean, std, eps]`. Enables reusing saved statistics. |
| `batch_size` | int | `4` | Batch size used by the DataLoader. |
| `num_workers` | int | `0` | Number of DataLoader subprocesses. Increase cautiously on macOS. |
| `split_fractions` | mapping | `{train: 0.7, val: 0.15, test: 0.15}` | Fractions used to partition sequential windows. Must sum to 1.0. |

### Additional Notes

- Splits are computed deterministically from the ordered time dimension and `stride`.
- When `mask_path` is provided, the mask is broadcast across time for normalisation and visualisation helpers.
- Register additional datasets using `neuralcora.dataset.register_dataset("my_key", MyDataset)` and update `dataset_name` accordingly.

## Model Block (`ModelConfig`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | `"simple_cnn"` | Model registry key resolved through `neuralcora.modelzoo.get_model_class`. |
| `input_channels` | int | `1` | Explicit input channel count. When set to 0/`null`, it falls back to `data.input_steps`. |
| `output_channels` | int | `6` | Explicit output channel count. Defaults to `data.forecast_steps` when unset. |
| `hidden_channels` | sequence[int] | `(32, 64, 64)` | Feature sizes for the sequential Conv2d blocks. |
| `kernel_size` | int | `3` | Kernel size used for each convolution (padding preserves spatial dimensions). |
| `dropout` | float | `0.0` | Dropout probability applied after ReLU activations. |

To introduce new architectures, add a module under `neuralcora/modelzoo`, register it via `register_model`, and set `model.name` to the new key.

## Training Block (`TrainingConfig`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `trainer_name` | str | `"basic"` | Trainer registry key. The bundled trainer uses Adam and MSE loss. |
| `epochs` | int | `1` | Number of training epochs. |
| `learning_rate` | float | `1e-3` | Optimiser learning rate. |
| `weight_decay` | float | `0.0` | L2 regularisation passed to Adam. |
| `device` | str | `"cpu"` | Target device. Set to `"auto"` to choose CUDA → MPS → CPU automatically. |
| `log_interval` | int | `10` | Steps between progress prints. |
| `save_dir` | str | `"runs"` | Root directory where trainer outputs are stored. |
| `gradient_clip_norm` | float or null | `null` | If provided, gradients are clipped to this norm. |

### Trainer Utilities

`BasicTrainer` records per-epoch metrics and exposes:

- `plot_history()` — plots training/validation MSE and RMSE (requires `matplotlib`).
- `visualize_validation_sample()` — animates predictions vs targets for any validation index.
- `evaluate()` / `predict()` — convenience wrappers for inference pipelines.

## Working with YAML Configs

- Example baseline config: `examples/01-Introduction/config.yml`.
- Launch training via CLI:
  ```bash
  python -m neuralcora.nh_run train --config examples/01-Introduction/config.yml
  ```
- Override values programmatically:
  ```python
  from neuralcora.utils import load_config
  cfg = load_config("examples/01-Introduction/config.yml", overrides={"training": {"epochs": 5}})
  ```

## Normalisation Workflow

Use `CoraDataModule.save_stats("path.pt")` after `setup(stage="fit")` to persist computed statistics. Later runs can call `load_stats("path.pt")` before creating validation/test loaders to ensure consistent scaling across experiments.

## Extensibility

- **Datasets** — implement a `torch.utils.data.Dataset`, register it, and update `data.dataset_name`.
- **Models** — author new architectures in `neuralcora/modelzoo` and register them under a unique key.
- **Trainers** — extend the trainer module with custom loops and expose them via `get_trainer_class`.

By standardising configs across these components, NeuralCora enables reproducible coastal forecasting experiments while staying close to the NeuralHydrology workflow.
