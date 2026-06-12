# PM2.5-Unet: Delhi PM2.5 Prediction Project

## Overview

This repository implements an end-to-end workflow for predicting PM2.5 in Delhi using a combination of:

- Station-level observations (OpenAQ)
- ERA5 meteorological reanalysis variables
- static urban/elevation features

The project includes:

- Reproducible data preparation
- Feature engineering and modeling notebooks
- XGBoost baselines and evaluation
- A Streamlit dashboard for interactive demonstration
- Presentation and report artifacts for project submission

## Project structure

### Core files

- `01_build_download_dataset.ipynb`  
  Builds and packages the Delhi data bundle.
- `02_train_xgboost_model.ipynb`  
  Main training and evaluation workflow (including ANN comparison logic and station-based experiments).
- `streamlit_app.py`  
  Interactive dashboard for browsing model outputs.
- `pm25_delhi_bundle/`  
  Input bundle consumed by the notebook and dashboard:
  - `stations_urban.csv`
  - `openaq_pm25.csv`
  - `era5_delhi.nc`
  - `era5_meteo.csv`
  - `stations_elevation.csv`
  - `data_manifest.json`
- `model_outputs/`  
  Saved outputs such as predictions, master tables, and feature-importance files.
- `fix_dataset_temp.py`  
  Utility script for repairing/rebuilding `openaq_pm25.csv` when needed.
- `requirements.txt`  
  Python dependencies for the project.
- `REPORT.md`  
  Concise technical summary.
- `PROJECT_STRUCTURE.md`  
  Compact list of active vs. parked resources.

### Presentations and reports

- `LY_mid_term_presentation/`
- `LY_final_term_presentation/`
- `LY_final_term_presentation/presentation/presentation.md`

### Parked extras

- `unused_or_external/` contains files not required for the active PM2.5 workflow.

## Data pipeline (high level)

1. Build the dataset bundle with `01_build_download_dataset.ipynb`.
2. Train and validate models in `02_train_xgboost_model.ipynb`.
3. Generate predictions and model summaries in `model_outputs/`.
4. Visualize and explain results with `streamlit_app.py` and project slides.

## Notebook workflow details

The training notebook currently covers:

- Temporal feature engineering:
  - `month`, `dayofweek`, sine/cosine cyclic encodings
  - meteorology and static feature integration
  - lag and rolling history features for PM2.5
- Target transformation (`log1p`) and inverse transform handling
- Two evaluation regimes:
  - LOSO (leave-one-station-out)
  - chronological holdout for single-station mode
- Baseline blending experiments (lag and XGBoost candidates)
- Selection of best model variant based on MAE

## Quick start

```bash
pip install -r requirements.txt
```

1. Open `01_build_download_dataset.ipynb` to refresh the bundle if needed.
2. Open `02_train_xgboost_model.ipynb` and run cells end-to-end.
3. Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

## Notes

- This repository has been cleaned to avoid pushing large binary files to Git history.
- Presentation assets are included in the `LY_*_term_presentation` folders for review/demo.

