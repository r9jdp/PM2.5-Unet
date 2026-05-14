# Mid-Term Presentation Slide Content Guide

Use this file as direct slide-writing content for your mid-term PPT.  
Scope is intentionally limited to: **data collection -> preprocessing -> XGBoost baseline -> dashboard**.

---

## Slide 1 - Title
**Title:**  
PM2.5 Estimation over Delhi-NCR: Mid-Term Progress Report

**Write on slide:**
- Department / course details
- Group details / guide name
- Presenter name
- Date

**One-line opening to speak:**  
"This mid-term presentation covers end-to-end implementation completed from authentic data collection to XGBoost baseline and dashboard."

---

## Slide 2 - Problem Context
**Title:**  
Why This Project Is Needed

**Write on slide:**
- Delhi-NCR has limited station coverage compared to full urban spatial variability.
- PM2.5 changes significantly across traffic zones, dense built-up regions, and seasonal weather patterns.
- Need a data-driven system that can estimate PM2.5 reliably using multi-source signals.
- Current station readings are point observations, while decisions need area-level understanding.
- Citizens and planners need spatially continuous PM insights, not only station-level numbers.
- A robust baseline is required first before moving to advanced spatial deep learning.

**Speak track:**  
"The core challenge is sparse station observations versus highly variable urban air pollution patterns."

---

## Slide 3 - Original Project Objective (From Initial Proposal)
**Title:**  
Target Vision of Project

**Write on slide:**
- Build a high-resolution PM2.5 estimation framework for Delhi-NCR.
- Integrate atmospheric, meteorological, and urban morphology inputs.
- Long-term direction includes deep spatial modeling (U-Net with attention).
- Design a reproducible pipeline where every data source is API-driven and versioned.
- Ensure the model can generalize across stations and time, not only memorized station behavior.
- Create a practical dashboard interface for technical and non-technical stakeholders.

**Speak track:**  
"The long-term research objective remains the same as our original proposal, but today I present the implemented baseline stage."

---

## Slide 4 - Mid-Term Scope (What Is Completed Now)
**Title:**  
Mid-Term Deliverables Completed

**Write on slide:**
- Automated data collection pipeline (API-based)
- Unified dataset bundle creation with metadata manifest
- Data cleaning + feature engineering pipeline
- XGBoost baseline training and evaluation
- Interactive dashboard for results and diagnostics
- Reproducible artifact generation for review (`predictions`, `feature_importance`, `master_table`)
- Quantitative evaluation on chronological holdout to avoid leakage

**Write on slide (important):**
- This mid-term does **not** include ANN/U-Net implementation discussion.
- ANN and U-Net are intentionally kept for next phase comparison and extension.

---

## Slide 5 - Data Sources and Authenticity
**Title:**  
Data Sources Used (Authentic External Sources)

**Write on slide:**
- PM2.5: OpenAQ API (daily station observations)
- Meteorology: ERA5-Land via CDS API
- Elevation: SRTM raster source
- Urban context: OSM-based neighborhood features (road/building density)
- Station framework: CPCB-style station mapping in Delhi region

**Write on slide (authenticity statement):**
- Data files are tracked in `data_manifest.json` with status metadata and file-level details.
- Data ingestion is script-driven, timestamped, and stored with row counts and file sizes.
- No manual editing workflow is used for core source files.

---

## Slide 6 - Data Collection Architecture
**Title:**  
How Data Is Collected Programmatically

**Write on slide as flow:**
1. Define station coordinates across Delhi bounding box  
2. Call OpenAQ API for PM2.5 measurements  
3. Call CDS API for ERA5-Land weather variables  
4. Extract elevation from SRTM raster  
5. Compute urban context features around stations  
6. Save all outputs into `pm25_delhi_bundle/` and zip

**Speak track:**  
"Collection is fully script/notebook driven, not manual copy-paste."

**Add on slide (small box):**
- Date window used: `2023-01-01` to `2024-12-31`
- Station framework used: 40 stations in Delhi bounding region

---

## Slide 7 - Dataset Bundle Snapshot
**Title:**  
Current Dataset Bundle (Mid-Term)

**Write on slide (table format):**
- `stations_urban.csv` -> station metadata + building/road density  
- `openaq_pm25.csv` -> station-date PM2.5 values  
- `era5_meteo.csv` -> station-date meteorology  
- `stations_elevation.csv` -> elevation by station  
- `era5_delhi.nc` -> raw ERA5 archive  
- `data_manifest.json` -> metadata + tracking

**Write on slide (key numbers):**
- 40 stations  
- Date coverage: 2023-01-01 to 2024-12-31  
- 29,240 station-date rows in PM2.5 and ERA5 tables

---

## Slide 8 - Data Processing and Cleaning
**Title:**  
Preprocessing Pipeline

**Write on slide:**
- Standardized station IDs, date formats, numeric types
- Harmonized ERA5 feature names to model-friendly schema
- Merged all sources into a station-date master table
- Missing-value treatment: station-level median fill followed by global median fallback
- Chronological sorting for time-dependent feature generation
- Conversion to training-safe numeric schema before model fit
- Created station-aware temporal structure for lag/rolling features

---

## Slide 9 - Feature Engineering
**Title:**  
Model Features Designed

**Write on slide in groups:**
- Temporal: month/day-of-week/day-of-year + cyclic encodings
- Weather: temp, precipitation, pressure, wind components/speed, dewpoint
- Spatial static: latitude, longitude, road density, building density, elevation
- PM history: lag1/lag2/lag3/lag7 + rolling mean/std features
- Target transform: `log1p(PM2.5)`

**Speak track:**  
"This feature space combines seasonality, meteorology, station context, and short-term PM memory."

---

## Slide 10 - Baseline Model Setup (XGBoost)
**Title:**  
XGBoost Baseline Design

**Write on slide:**
- Model: `XGBRegressor` (regression)
- Train/test protocol: chronological 80/20 split per station
- Leakage prevention: train on earlier dates, test on later dates
- Additional baseline context:
  - station-month climatology computed from train split
  - blend optimization among XGBoost, lag baseline, climatology

**Write on slide (parameters summary):**
- `n_estimators=1200`, `max_depth=6`, `learning_rate=0.03`, subsampling + regularization
- Target transformation: train on `log1p(PM2.5)` and invert using `expm1` at inference
- Final prediction from optimized blend of model + lag + climatology

---

## Slide 11 - Performance Results (Current Run)
**Title:**  
XGBoost Mid-Term Results

**Write on slide (big numbers):**
- Test rows: 5,880
- Stations in evaluation: 40
- Test period: 2024-08-07 to 2024-12-31

**Write on slide (metrics):**
- RMSE: **21.37**
- MAE: **16.55**
- R2: **0.9333**
- MAPE: **25.94%**
- sMAPE: **19.15%**
- Bias: **-0.98**

**Speak track:**  
"The current baseline shows strong temporal holdout performance with high R2 and low bias."

**Add interpretation line on slide:**
- Model captures temporal variation well across stations; residual bias is near zero.
- Error metrics are reported in PM2.5 concentration units and percent forms.

---

## Slide 12 - Feature Importance
**Title:**  
What Drives Predictions Most

**Write on slide (Top features):**
1. `clim_station_month`
2. `month_sin`
3. `month_cos`
4. `pm25_roll3_mean`
5. `pm25_roll7_mean`

**Interpretation text for slide:**
- Seasonal behavior and station climatology dominate.
- Recent PM trend features are also strong contributors.
- This is consistent with Delhi’s strong seasonal pollution pattern and station-specific behavior.

---

## Slide 13 - Dashboard for Faculty Demonstration
**Title:**  
Interactive Dashboard (Streamlit)

**Write on slide:**
- KPI cards (MAE, RMSE, R2, sMAPE, row/station counts)
- Actual vs predicted trend visualization
- Parity and residual diagnostic plots
- Worst-error day table
- Feature-importance chart
- Downloadable CSV outputs for evidence

**Speak track:**  
"Dashboard provides both communication and technical validation in one interface."

**Add one line on slide:**
- Faculty can review both summary KPIs and diagnostic plots from the same run outputs.

---

## Slide 14 - End-to-End Output Artifacts
**Title:**  
Artifacts Generated So Far

**Write on slide:**
- `pm25_delhi_bundle/` (processed data package)
- `model_outputs/predictions_all.csv`
- `model_outputs/feature_importance.csv`
- `model_outputs/delhi_pm25_master.csv`
- `streamlit_app.py`

**Write on slide:**
- These files make the mid-term workflow reproducible and reviewable.
- The same outputs are used by both technical analysis and dashboard presentation layers.

---

## Slide 15 - Mid-Term Conclusion
**Title:**  
What Is Completed at Mid-Term

**Write on slide:**
- Data pipeline established from collection to storage
- Multi-source data integrated and cleaned
- Strong XGBoost baseline implemented and validated
- Visualization and reporting layer delivered

**Closing line to speak:**  
"I have completed the complete baseline phase and established a reliable foundation for model comparison in the next phase."

**Add on slide (final takeaway):**
- Mid-term objective achieved: reliable pipeline + validated baseline + communication-ready interface.

---

## Slide 16 - Next Phase (Without Detailing Implementation)
**Title:**  
Post Mid-Term Roadmap

**Write on slide:**
- Fine-tune and evaluate ANN regressor on same split/metrics protocol
- Compare ANN vs XGBoost performance fairly
- Then train U-Net model for PM2.5 estimation
- Extend dashboard with model-wise comparison views

**Important:**  
Keep this slide future-oriented only. Do not present ANN implementation details in mid-term.
- Keep focus on fair comparison protocol and planned expansion, not on implementation internals.

---

## Slide 17 - Q&A / Thank You
**Title:**  
Thank You

**Write on slide:**
- "Questions and feedback"
- Contact / group details (optional)
- "I can show live dashboard diagnostics if required."

---

## Optional Backup Slides (If Faculty Asks)
You can keep 2 backup slides ready:

1. **Data Schema Backup**
- Column-level overview of each CSV file

2. **Modeling Protocol Backup**
- Exact train/test split logic
- Why chronological split was chosen

---

## Suggested Presentation Order and Timing (10-12 min)
- Slides 1-4: 2.0 min
- Slides 5-9: 3.5 min
- Slides 10-12: 2.5 min
- Slides 13-16: 2.5 min
- Slide 17 + Q&A: remaining time

This keeps the narrative focused, technical, and mid-term appropriate.
