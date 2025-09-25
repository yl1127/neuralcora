# NeuralCora

NeuralCora is a PyTorch-based research framework for CNN-driven coastal zeta forecasting inspired by NeuralHydrology. The initial release focuses on the CORA dataset distributed with this repository and demonstrates an end-to-end workflow (data loading, model training, evaluation) configured via YAML-style experiment definitions.

## Quickstart
1. Install in editable mode:
   ```bash
   pip install -e .[dev]
   ```
2. Launch the introduction notebook at `examples/01-Introduction/Introduction.ipynb` to preview the data pipeline and baseline model.
3. Explore the configuration utilities in `neuralcora/utils/config.py` to author custom experiments.

## Repository Layout
- `neuralcora/`: Python package implementing datasets, models, training logic, and evaluation helpers.
- `data/`: Sample CORA NetCDF cubes used in the demo.
- `docs/`: Project documentation, including requirements.
- `examples/`: Jupyter notebooks demonstrating library usage.

## Requirements
Core dependencies are declared in `pyproject.toml` and include PyTorch, NumPy, Xarray, and NetCDF4. Optional extras for development and docs are provided as well.

## License
TBD.
