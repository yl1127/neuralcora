# NeuralCora Data & Metrics Requirements

## 1. Project Purpose
- Deliver a PyTorch/CNN-based research framework for coastal zeta (sea-surface elevation) forecasting over the US East Coast, with an initial focus on the CORA-derived samples shipped under `data/`.
- Enable reproducible experiments: keep configuration-driven pipelines, versioned datasets, and automated evaluation similar to NeuralHydrology.

## 2. Target Use Cases
- Sub-daily to multi-day zeta forecasts on a fixed coastal grid to support operational surge monitoring and research.
- Data assimilation experiments that blend observational grids (e.g., tide gauges, altimetry) with numerical weather prediction forcings.
- Sensitivity studies evaluating coastal dynamics drivers (wind, pressure, bathymetry) via explainable CNN modules.

## 3. Datasets

### 3.1 Primary (In-Repository Prototype)
| Dataset | Path | Status | Spatial/Temporal Resolution | Key Variables | Intended Usage |
|---------|------|--------|-----------------------------|---------------|----------------|
| CORA coastal sample | `data/NY_2013_180_360.nc` | Provided | 180×360 lat/lon grid, hourly cadence for full 2013 year (8760 steps) | `zeta(time, latitude, longitude)`; coordinate arrays `time`, `latitude`, `longitude` | Core training/validation target for NeuralCora CNNs |
| Land/sea mask | `data/real_land_mask_180_360.nc` | Provided | 180×360 static mask aligned with CORA grid | Binary land/ocean; extend with distance-to-coast if needed | Spatial masking, loss weighting |

Confirmed metadata snapshot (NetCDF):
```
xarray.Dataset {
  dimensions:
    time = 8760
    latitude = 180
    longitude = 360
  variables:
    float64 zeta(time, latitude, longitude)
    datetime64[ns] time(time)
      : long_name = "model time"
      : standard_name = "time"
    float64 latitude(latitude)
    float64 longitude(longitude)
}
```
Open validation task: inspect for units, missing-value flags, and quality-control attributes prior to integration.

### 3.2 External Forcing & Attribute Candidates
| Purpose | Candidate Source | Notes |
|---------|------------------|-------|
| (Deferred for MVP) Atmospheric forcings (wind, pressure) | ERA5, HRRR (NOAA) | Regrid to 180×360; align timestamps; licence compatible (Copernicus/NOAA). |
| (Deferred for MVP) Hydrological runoff | USGS, NWM | Optional driver for estuary influence; may require coarse-to-fine interpolation. |
| (Deferred for MVP) Sea-level anomalies / tides | NOAA CO-OPS tide gauges, AVISO satellite altimetry | Assimilate point observations via rasterization encoder. |
| Static bathymetry / coastline | GEBCO, NOAA ETOPO | Derive slope, depth features for conditioning CNN (future milestone). |
| Storm tracks | IBTrACS | Provide polygon masks or feature channels for extreme-event training (future milestone). |

Initial release focuses exclusively on the `zeta` field; external forcings are parked for subsequent phases.

### 3.3 Data Governance Requirements
- Maintain a `datasets/README.md` describing acquisition scripts, licences, citation requirements, and preprocessing steps for each source.
- Use declarative dataset configs (YAML) to record paths, expected variable names, normalization schemes, and masks per experiment.
- Adopt DVC or alternative versioning for large external datasets; store access tokens outside repo.

## 4. Data Preprocessing Deliverables
- Scripts/notebooks to inspect NetCDF metadata and export summary statistics (mean, std, missing fractions, units).
- Consistent resampling to canonical prediction cadence (hourly baseline). Document any interpolation for alternative cadences.
- Normalization pipelines with global statistics as default; allow config switch to region-specific stats when warranted.
- Train/validation/test split definitions (e.g., by year, storm events) stored in reproducible index files.
- Patch extraction utilities (optional) for localized CNN training.

## 5. Evaluation Metrics & Diagnostics

### 5.1 Core Continuous Metrics
- RMSE, MAE, Mean Bias for zeta across spatial grid and aggregated over coastal segments.
- Normalized Root Mean Square Error (NRMSE) using observed range per grid point.
- Correlation coefficient (Pearson) for temporal alignment at each location.

### 5.2 Extreme Event & Threshold Metrics
- Peak error metrics: max surge height error, time-of-peak offset (hours).
- Threshold-based hit/false-alarm/miss rates for flood levels (configurable thresholds; defaults unset for MVP).
- Event-based continuous ranked probability score (CRPS) when probabilistic outputs are enabled.

### 5.3 Spatial Structure Metrics (CNN-Specific)
- Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) on 2D zeta snapshots.
- Earth Mover's Distance (EMD) or Wasserstein distance for comparing spatial distributions of surge.
- Gradient difference loss (GDL) style metrics to assess shoreline gradient accuracy.

### 5.4 Temporal Consistency Metrics
- Pinball loss / CRPS for quantile forecasts (if modeled).
- Spectral error metrics (Fourier-based) to capture tidal frequency fidelity.
- Autocorrelation function deviation vs. observations for lead-time stability.

### 5.5 Operational Diagnostics & Visuals
- Basin-/segment-level scorecards summarizing above metrics per coastal region (NY Harbor, Long Island Sound, etc.).
- Animated or static map plots comparing predicted vs. observed zeta at key timestamps.
- Hydrograph-style plots for representative coastal points aligned with CO-OPS stations.

## 6. Reporting & Benchmarking
- Standard evaluation pipeline triggered via `python -m neuralcora.run evaluate --config <yaml>` producing metrics CSV, NetCDF outputs, and plots.
- Template Jupyter notebooks inside `examples/` demonstrating dataset load, preprocessing checks, and metric interpretation.
- Baseline CNN benchmark definitions (e.g., UNet, ResNet encoder-decoder) with documented hyperparameters and metrics to track progress.

## 7. Decisions to Operationalize
1. Implement hourly "1-step in, 6-step out" forecast horizon as configurable defaults (include config keys for alternative lead times and receptive field length).
2. Encode global normalization statistics loader/saver; allow optional overrides per coastal subregion.
3. Build metadata inspection script to surface units, missing flags, and coordinate ranges within CI or preprocessing pipeline.
4. Plan roadmap milestone for integrating first external forcing dataset (likely ERA5 winds/pressure) once MVP stabilizes.
5. Define flood-threshold catalog and regulatory reporting templates when engaging with stakeholders; keep placeholders in evaluation pipeline.
