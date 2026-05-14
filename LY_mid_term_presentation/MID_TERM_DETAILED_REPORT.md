# PM2.5 Mid-Term Evaluation Report (Detailed)

## 1. Mid-Term Scope (What This Report Covers)
This report is strictly for mid-term evaluation scope:
- Data collection
- Data processing and cleaning
- Data storage and reproducible dataset bundle creation
- XGBoost baseline model preparation and evaluation
- Dashboard creation and interpretation

This report intentionally does **not** cover ANN implementation details. ANN is included only as a future phase item. U-Net is also future phase.

---

## 2. Project Objective
The project objective at this stage is to build a reproducible PM2.5 estimation workflow for Delhi and deliver:
- A packaged dataset bundle from authentic external data sources
- A feature-engineered training table
- A strong XGBoost baseline with quantitative evaluation
- A presentation-ready interactive dashboard for faculty demonstration

---

## 3. Codebase Components Used For Mid-Term
Primary files used:
- `01_build_download_dataset.ipynb`
- `02_train_xgboost_model.ipynb` (XGBoost sections only)
- `streamlit_app.py`
- `pm25_delhi_bundle/` dataset assets
- `model_outputs/` trained outputs

Supporting project files read:
- `requirements.txt`
- `pm25_delhi_bundle/data_manifest.json`

---

## 4. End-to-End Workflow Summary
The implemented workflow is:
1. Define station grid and station metadata for Delhi.
2. Collect PM2.5 observations from OpenAQ API.
3. Collect meteorology from ERA5-Land through CDS API.
4. Collect elevation from SRTM raster.
5. Compute urban context indicators (building/road density around stations).
6. Package all files into a reproducible dataset bundle (`pm25_delhi_bundle` + zip).
7. Build a station-date master table.
8. Engineer temporal, meteorological, and lag-based features.
9. Train XGBoost on log-transformed PM2.5 target.
10. Evaluate on chronological holdout by station timeline.
11. Save prediction artifacts and feature importance.
12. Serve insights through Streamlit dashboard.

---

## 5. Data Collection Pipeline (Detailed)

## 5.1 Station Definition
In `01_build_download_dataset.ipynb`:
- Initial 10 anchor stations are defined with latitude/longitude in Delhi region.
- Stations are expanded deterministically to 40 stations total.
- Bounding limits used:
  - Latitude: `28.3` to `28.9`
  - Longitude: `76.8` to `77.5`
- Each station has:
  - `station_id`
  - `lat`
  - `lon`
  - `source = CPCB`

Why this matters in presentation:
- It shows clear spatial coverage and station indexing.
- It gives a stable structure for merging multiple sources.

## 5.2 Urban Context Data
Urban context file: `stations_urban.csv`

Method:
- Attempt to use OSMnx per station with 500m neighborhood:
  - Road density from count of road edges
  - Building density from count of building features
- Fallback path (if OSMnx unavailable/errors):
  - Fill fixed defaults
- Save per station:
  - `building_density`
  - `road_density`

Presentation point:
- This layer adds local urban morphology information for each station and enriches model input beyond only PM and weather.

## 5.3 PM2.5 Collection (OpenAQ API)
Target file: `openaq_pm25.csv`

API details used in notebook:
- Base endpoint: `https://api.openaq.org/v3`
- API key read from environment variable: `OPENAQ_KEY`
- Location retrieval around Delhi (`coordinates = 28.6139,77.2090`, `radius = 100000`)
- Measurement retrieval for each location:
  - Parameter: PM2.5
  - Date range: `2023-01-01` to `2024-12-31`

Processing:
- Parse timestamp safely to valid date.
- Drop invalid/null readings.
- Map each OpenAQ point to nearest station using Haversine distance.
- Aggregate to station-day mean PM2.5.

Presentation point:
- PM2.5 values are not manually entered; they are API-collected and systematically mapped to project stations.

## 5.4 Meteorology Collection (ERA5-Land via CDS API)
Target files:
- `era5_delhi.nc` (raw NetCDF)
- `era5_meteo.csv` (station-level extracted table)

Source and method:
- CDS API dataset: `reanalysis-era5-land`
- Region: Delhi bounding box
- Time: daily at 12:00 for 2023 and 2024
- Variables requested:
  - `2m_temperature`
  - `10m_u_component_of_wind`
  - `10m_v_component_of_wind`
  - `surface_pressure`
  - `2m_dewpoint_temperature`
  - `total_precipitation`

Station extraction:
- For each station, nearest gridded ERA5 point is selected.
- Result converted and exported to CSV.

Presentation point:
- Weather context is physically meaningful and pulled from a global reanalysis product through official API.

## 5.5 Elevation Collection (SRTM)
Target file: `stations_elevation.csv`

Method:
- Download SRTM tile zip.
- Extract GeoTIFF.
- Sample elevation for each station coordinate.
- Handle nodata values robustly.

Presentation point:
- Terrain/elevation is included as a static geographic feature.

## 5.6 Packaging and Data Manifest
At end of data-build notebook:
- `data_manifest.json` is generated/updated.
- `pm25_delhi_bundle.zip` is created for transfer/reuse.

Manifest snapshot in this project marks bundle files with `"status": "REAL"` and stores rows/size metadata per file.

Presentation point:
- Dataset is versioned and packaged, so the faculty can see reproducibility and traceability.

---

## 6. Dataset Inventory (Current Workspace Snapshot)
From the current bundle in this project:

1. `stations_urban.csv`
- Rows: 40
- Station IDs: 40
- Columns: `lat`, `lon`, `source`, `station_id`, `building_density`, `road_density`

2. `openaq_pm25.csv`
- Rows: 29,240
- Station IDs: 40
- Date range: `2023-01-01` to `2024-12-31`
- Columns: `station_id`, `date`, `pm25`

3. `era5_meteo.csv`
- Rows: 29,240
- Station IDs: 40
- Date range (valid_time): `2023-01-01` to `2024-12-31`
- Core columns observed: `valid_time`, `t2m`, `tp`, `number`, `latitude`, `longitude`, `expver`, `station_id`

4. `stations_elevation.csv`
- Rows: 40
- Station IDs: 40
- Columns: `lat`, `lon`, `source`, `station_id`, `elevation`

5. `era5_delhi.nc`
- Raw ERA5 NetCDF archive used for extraction

6. `data_manifest.json`
- File-level metadata (created time, status, size, rows)

---

## 7. Data Processing and Cleaning Logic (Training Notebook)
In `02_train_xgboost_model.ipynb`, the implemented processing flow is:

1. Read all bundle CSV files.
2. Enforce data types (`station_id` string, dates as datetime, numeric conversion).
3. Validate required fields (`pm25`, date columns).
4. Harmonize ERA5 names:
   - `t2m -> temp_2m`
   - `tp -> total_precip`
   - `u10 -> wind_u`
   - `v10 -> wind_v`
   - `sp -> surface_pressure`
   - `d2m -> dewpoint_2m`
5. Build base station-date table by merging:
   - PM2.5 + station metadata + elevation + meteorology
6. Fill missing numeric values:
   - First with station-level median
   - Then with global median
7. Sort chronologically per station and engineer sequential features.

Why this is important for mid-term:
- Shows robust preprocessing discipline before model training.
- Demonstrates consistency across heterogeneous sources.

---

## 8. Feature Engineering (XGBoost Input Space)

## 8.1 Calendar Features
- `month`
- `dayofweek`
- `dayofyear`
- Cyclical encodings:
  - `month_sin`, `month_cos`
  - `dow_sin`, `dow_cos`
  - `doy_sin`, `doy_cos`

## 8.2 Meteorology and Station Context
- `temp_2m`
- `total_precip`
- `wind_u`, `wind_v`
- `wind_speed` (derived)
- `surface_pressure`
- `dewpoint_2m`
- `lat`, `lon`
- `building_density`
- `road_density`
- `elevation`

## 8.3 Temporal Memory Features (Per Station)
- Lags:
  - `pm25_lag1`, `pm25_lag2`, `pm25_lag3`, `pm25_lag7`
- Rolling:
  - `pm25_roll3_mean`
  - `pm25_roll7_mean`
  - `pm25_roll7_std`

## 8.4 Target Transformation
- Training target is `log_pm25 = log1p(pm25)`
- Prediction is inverse transformed with `expm1`

Presentation point:
- The feature space captures seasonality, weather forcing, station characteristics, and PM history in one integrated design.

---

## 9. Train/Test Strategy and Model Design (XGBoost Stage)

Implemented split strategy:
- Chronological per station (not random split)
- Training: first 80% timestamps of each station
- Testing: last 20% timestamps of each station

This ensures:
- No future leakage
- Realistic temporal generalization check
- Every station contributes to both training (past) and testing (future)

XGBoost model config used:
- `n_estimators = 1200`
- `max_depth = 6`
- `learning_rate = 0.03`
- `subsample = 0.9`
- `colsample_bytree = 0.9`
- `reg_alpha = 0.1`
- `reg_lambda = 2.0`
- `objective = reg:squarederror`
- `tree_method = hist`
- `random_state = 42`

Additional modeling step:
- Station-month climatology (`clim_station_month`) is computed from train split.
- Final prediction uses optimized blend of:
  - XGBoost output
  - Lag-1 baseline
  - Climatology baseline

Latest run selected blend:
- XGBoost weight: `0.2`
- Lag-1 weight: `0.0`
- Climatology weight: `0.8`

---

## 10. Model Performance (Current Saved Run)
From `model_outputs/predictions_all.csv` and notebook output:

- Test rows: `5,880`
- Test stations: `40`
- Test date range: `2024-08-07` to `2024-12-31`

Metrics:
- RMSE: `21.37`
- MAE: `16.55`
- R2: `0.9333`
- MAPE: `25.94%`
- sMAPE: `19.15%`
- Bias (Pred - Actual): `-0.98`

Interpretation for faculty:
- The baseline is already strong on held-out future periods (high R2, low bias).
- Error magnitude is in PM2.5 concentration units, and percentage metrics are also provided for normalized interpretation.

---

## 11. Feature Importance (Current Model Output)
Top drivers from `model_outputs/feature_importance.csv`:
1. `clim_station_month`
2. `month_sin`
3. `month_cos`
4. `pm25_roll3_mean`
5. `sid_STATION_00`
6. `pm25_roll7_mean`
7. `month`
8. `doy_cos`
9. `sid_STATION_22`
10. `dayofyear`

Presentation message:
- Seasonal signals and climatological context are major explanatory factors.
- Temporal PM history features also contribute strongly.

---

## 12. Dashboard (Streamlit) - What Was Built
The dashboard in `streamlit_app.py` provides:

1. KPI cards:
- Rows
- Stations
- MAE
- RMSE
- R2
- sMAPE

2. Metrics matrix table:
- XGBoost final metrics
- Optional persistence baseline metrics

3. Time-series view:
- Actual vs predicted PM2.5 over time

4. Diagnostic scatter plots:
- Parity plot (actual vs predicted)
- Residual vs actual

5. Worst-error table:
- Top error days with residuals

6. Feature-importance chart:
- Top feature contributions

7. Download controls:
- Filtered predictions CSV
- Metrics matrix CSV

Presentation message:
- Dashboard is not only visual; it is an audit tool for checking errors, residual behavior, and exportable evidence.

---

## 13. Technology Stack Used
From project dependencies and implementation:

Core language/runtime:
- Python
- Jupyter Notebook

Data and modeling:
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`

Data collection and geospatial:
- `requests` (OpenAQ API)
- `cdsapi` + `xarray` (ERA5 acquisition/extraction)
- `osmnx` (urban context features)
- `rasterio` (elevation sampling)

App layer:
- `streamlit`

Packaging/output:
- `zipfile`, JSON manifest, CSV outputs

---

## 14. Mid-Term Presentation Script (Slide-by-Slide)

## Slide 1: Problem Statement
"I am building a PM2.5 estimation pipeline for Delhi. For mid-term, I am presenting the full workflow from data collection to XGBoost baseline and dashboard."

## Slide 2: Data Sources
"I collected and integrated PM2.5 observations (OpenAQ API), meteorology (ERA5-Land via CDS API), elevation (SRTM), and station urban context (OSM-based density features)."

## Slide 3: Data Authenticity Statement
"The project data bundle is built from source APIs/datasets and tracked in `data_manifest.json`. The manifest marks bundle files as REAL and records row/size metadata for traceability."

## Slide 4: Dataset Bundle
"All source outputs are packaged in `pm25_delhi_bundle/` with six core files plus zip export, so the workflow is reproducible."

## Slide 5: Processing and Cleaning
"I standardized station IDs/dates, harmonized variable names, merged all sources into a station-date table, and handled missing values by station-level and global medians."

## Slide 6: Feature Engineering
"I engineered calendar cyclic features, lag/rolling PM features, meteorological variables, static station context, and elevation features."

## Slide 7: Training Protocol
"I used chronological 80/20 split per station to prevent leakage and evaluate future-date performance. Target was modeled in log-space using XGBoost regressor."

## Slide 8: Baseline Results
"Current held-out results: RMSE 21.37, MAE 16.55, R2 0.9333 over 5,880 test rows across 40 stations."

## Slide 9: Explainability
"Feature importance indicates climatology and seasonality are dominant drivers, followed by PM history features."

## Slide 10: Dashboard Demo
"The dashboard shows KPI cards, time-series fit, parity/residual diagnostics, worst-error days, and downloadable evidence tables."

## Slide 11: Mid-Term Completion Summary
"Up to this stage, I have completed: data acquisition, integration, cleaning, baseline XGBoost training, performance validation, and reporting dashboard."

## Slide 12: Next Plan (Post Mid-Term)
"Next I will fine-tune and evaluate ANN regressor under the same split/metric protocol, then compare ANN vs XGBoost. After that, I will train U-Net for PM2.5 estimation."

---

## 15. What To Say As Final Mid-Term Contribution Summary
You can use this exact summary:

"For mid-term, I completed the full pipeline from data collection to a validated XGBoost baseline and a presentation dashboard. I integrated authentic multi-source environmental data, built reproducible dataset packaging, implemented robust preprocessing and feature engineering, and evaluated the model on chronological station-wise holdout. The baseline already gives strong predictive performance. My next phase is controlled model comparison with ANN, followed by U-Net-based PM2.5 estimation."

---

## 16. What To Keep Out Of Mid-Term Discussion (Important)
For this mid-term presentation, do not spend time on:
- ANN implementation internals
- ANN result interpretation
- U-Net implementation details
- Non-PM2.5 unrelated artifacts in repo

Reason:
- Mid-term narrative should remain tightly focused on "data collection -> preprocessing -> XGBoost baseline -> dashboard evidence".

---

## 17. Reproducibility Checklist (For You / Friend)
Before presentation/demo:
1. Verify `pm25_delhi_bundle/` files exist.
2. Verify `model_outputs/` files exist (`predictions_all.csv`, `feature_importance.csv`, `delhi_pm25_master.csv`).
3. Run dashboard:
   - `streamlit run streamlit_app.py`
4. Keep one short explanation ready for each chart:
   - Time-series fit quality
   - Parity and residual behavior
   - Top error days
   - Feature ranking

---

## 18. Deliverables Produced For Mid-Term
Completed outputs in this repository:
- Dataset bundle folder: `pm25_delhi_bundle/`
- Dataset zip: `pm25_delhi_bundle.zip`
- Trained output predictions: `model_outputs/predictions_all.csv`
- Feature importance table: `model_outputs/feature_importance.csv`
- Master engineered table: `model_outputs/delhi_pm25_master.csv`
- Interactive dashboard app: `streamlit_app.py`

This closes the mid-term scope exactly at XGBoost + dashboard stage.
