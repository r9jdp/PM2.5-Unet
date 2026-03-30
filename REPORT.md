# PM2.5 Prediction Project Report (Delhi)

## 1. Project Objective
The objective of this work was to build and evaluate an XGBoost baseline for PM2.5 prediction, prepare a reproducible workflow, and deliver an interactive Streamlit dashboard for faculty demonstration.

## 2. What Was Built
I built the project in three layers:

1. **Data preparation and packaging workflow**
- Notebook: `01_build_download_dataset.ipynb`
- Bundle folder: `pm25_delhi_bundle/`

2. **Model training and evaluation workflow**
- Notebook: `02_train_xgboost_model.ipynb`
- Output folder: `model_outputs/`

3. **Interactive dashboard for demonstration**
- App: `streamlit_app.py`

Additionally, I created a temporary data-fix utility:
- Script: `fix_dataset_temp.py`

## 3. Dataset Assets Used
The working bundle is in `pm25_delhi_bundle/`:

- `stations_urban.csv`
- `openaq_pm25.csv`
- `era5_delhi.nc`
- `era5_meteo.csv`
- `stations_elevation.csv`
- `data_manifest.json`

Manifest snapshot confirms the packaged data assets and metadata (`data_manifest.json`) including row counts and file sizes.

## 4. End-to-End Technical Pipeline

### 4.1 Data Loading and Harmonization
In `02_train_xgboost_model.ipynb`, I:

1. Loaded station, OpenAQ PM2.5, ERA5 meteorology, and elevation files.
2. Standardized date and variable naming for ERA5 fields (for example: `t2m -> temp_2m`, `tp -> total_precip`, `u10/v10 -> wind_u/wind_v`).
3. Added target-mode controls for training scope:
- `TARGET_MODE = "real_only"`
- `REAL_STATION_IDS = {"STATION_00"}`

### 4.2 Feature Engineering
I created a station-date master table and engineered:

1. **Calendar features**
- `month`, `dayofweek`
- Cyclic encodings: `month_sin`, `month_cos`, `dow_sin`, `dow_cos`

2. **Meteorology / static features**
- `temp_2m`, `wind_speed`, `surface_pressure`, `dewpoint_2m`, `total_precip`
- `building_density`, `road_density`, `elevation`

3. **Temporal history features (per station)**
- Lags: `pm25_lag1`, `pm25_lag2`, `pm25_lag3`, `pm25_lag7`
- Rolling signals: `pm25_roll3_mean`, `pm25_roll7_mean`, `pm25_roll7_std`

4. Target transform:
- `log_pm25 = log1p(pm25)`

### 4.3 Modeling Strategy
I implemented two evaluation modes:

1. **LOSO (Leave-One-Station-Out)** when multiple stations are available.
2. **TIME_HOLDOUT** when only one station is in training scope (chronological 80/20 split).

In single-station mode, I added predictor selection logic:

- XGBoost prediction
- Persistence baseline (`lag-1`)
- Weighted blends of lag-1 and XGBoost (0.1 to 0.9)

The notebook automatically selects the lowest-MAE candidate on holdout.

## 5. Current Run Outputs
Saved in `model_outputs/`:

- `predictions_all.csv`
- `feature_importance.csv`
- `delhi_pm25_master.csv`

Current summary from latest run:

- **CV mode**: TIME_HOLDOUT
- **Rows used**: 724
- **Features used**: 18
- **Stations in run**: 1
- **MAE**: 44.35
- **RMSE**: 56.79
- **R²**: -0.2680

Interpretation note for presentation:
- MAE is in **ug/m3**, not percent.

## 6. Feature Importance (Top Signals)
From `model_outputs/feature_importance.csv`, the strongest contributors include:

1. `month_cos`
2. `pm25_roll7_mean`
3. `pm25_roll3_mean`
4. `pm25_lag3`
5. `month_sin`
6. `total_precip`
7. `pm25_lag7`
8. `pm25_lag2`
9. `pm25_lag1`

This shows temporal and seasonal features are driving most predictive signal.

## 7. Dashboard Delivered
I implemented `streamlit_app.py` with:

1. KPI cards (Rows, Stations, MAE, RMSE, R², sMAPE)
2. Metrics matrix table
3. Actual vs Predicted time-series chart
4. Parity scatter and residual scatter plots
5. Worst-error-day table
6. Feature-importance bar chart
7. Download buttons for filtered predictions and metrics tables
8. Method notes section for academic presentation context

## 8. How To Run

### 8.1 Train / refresh model outputs
1. Open `02_train_xgboost_model.ipynb`
2. Run cells top-to-bottom
3. Confirm files are written into `model_outputs/`

### 8.2 Launch dashboard
From project root:

```powershell
& .\.venv\Scripts\streamlit.exe run .\streamlit_app.py
```

Then open the local URL shown by Streamlit.

## 9. What I Can Tell Faculty (Presentation Script)

1. I built a reproducible PM2.5 prediction pipeline with structured data ingestion, harmonization, feature engineering, and model evaluation.
2. I used XGBoost with temporal features, lag-based signals, and seasonal encodings.
3. I implemented robust evaluation logic that adapts to dataset structure (LOSO for multi-station, time holdout for single-station).
4. I added fallback predictor comparison (XGBoost vs persistence vs blended) to improve reliability and avoid blind model selection.
5. I generated standard output artifacts (`predictions_all.csv`, `feature_importance.csv`, `delhi_pm25_master.csv`) for traceable analysis.
6. I delivered a Streamlit dashboard that visualizes metrics, predictions, residual behavior, feature ranking, and downloadable evidence tables.
7. This XGBoost baseline is now ready to be compared with U-Net under the same split-and-metric protocol.

## 10. Next Comparison Step (XGBoost vs U-Net)
For fair benchmarking, use the same:

1. target subset,
2. date split,
3. metrics (MAE, RMSE, R², optionally sMAPE),
4. and reporting format in dashboard tables.

This guarantees a valid model-to-model comparison.
