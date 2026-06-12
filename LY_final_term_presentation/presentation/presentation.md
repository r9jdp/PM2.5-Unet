# Final-Term Evaluation Presentation Content

Use this file as the main content guide for the final-term PPT.  
The goal of this deck is to clearly show:

1. the complete project objective,
2. how the dataset was collected and prepared,
3. how the final semester implementation was carried out,
4. how XGBoost and ANN were implemented fairly,
5. and why ANN is slightly better in the final comparison.

The deck below is designed for a strong 12-15 minute faculty presentation.  
Each slide includes:

- title,
- slide content,
- presentation focus,
- image(s) to add from `presentation/images/`.

---

## Slide 1 - Title rajdeepp

**Title:** 
PM$_{2.5}$ Estimation over Delhi-NCR: Final-Term Evaluation

**Write on slide:**
- Final-Term Evaluation Report
- Group 38
- Rajdeep Pandey - 16014223064
- Vruddhi Mule - 16014223099
- Sagar Jadhav - 16014223070
- Sohom Mallick - 16014223083
- Third Year, AIDS
- KJ Somaiya College of Engineering
- Faculty Guide: Dr. Suchitra Patil

**What to say:**
This presentation summarizes the complete work done during the semester for PM$_{2.5}$ estimation over Delhi-NCR. It includes the full data pipeline, model-building workflow, final ANN vs XGBoost comparison, and the final results obtained on the test set.

**Image to add:**
- Optional: no image needed, or use a clean title background only

---

## Slide 2 - Problem Context
**Title:**  
Why This Project Is Important

**Write on slide:**
- Delhi-NCR experiences severe and highly variable PM$_{2.5}$ pollution.
- PM$_{2.5}$ changes due to season, weather, traffic, and local urban conditions.
- Observed station data are point-based and do not directly provide a complete predictive understanding.
- A reproducible data-driven system is required to estimate PM$_{2.5}$ using collected environmental signals.
- The problem is not only forecasting but also integrating multi-source environmental information into one modeling framework.

**What to say:**
The main challenge is that PM$_{2.5}$ is not stable over time. It changes strongly across seasons and also fluctuates day to day. Therefore, we need a system that uses pollutant observations together with meteorological and contextual data to produce reliable estimates.

**Image to add:**
- `slide02_pm25_context_trend.png`

---

## Slide 3 - Project Objective
**Title:**  
Project Objective

**Write on slide:**
- Build an end-to-end PM$_{2.5}$ estimation framework for Delhi-NCR.
- Collect PM$_{2.5}$, meteorological, elevation, and urban-context data from authentic external sources.
- Prepare a unified station-date dataset for supervised environmental learning.
- Engineer temporal, seasonal, historical, and contextual features.
- Implement XGBoost as a strong machine learning baseline.
- Implement ANN for nonlinear predictive comparison on the same feature space.
- Extend the project direction toward deeper spatial modeling through U-Net based PM$_{2.5}$ estimation.
- Present the complete workflow through a technical report and a faculty-facing dashboard.

**What to say:**
The project objective is broader than one model result. It includes collected-data integration, feature engineering, implementation of XGBoost and ANN, fair comparative evaluation, and the longer-term direction toward U-Net based deep spatial modeling.

**Image to add:**
- Optional: no image required

---

## Slide 4 - Final Semester Implementation
**Title:**  
Final Semester Implementation

**Write on slide:**
- collected multi-source dataset prepared and organized into a station-date master table
- preprocessing and harmonization completed for PM$_{2.5}$ and meteorological variables
- temporal, lag-based, rolling, and seasonal features engineered
- XGBoost baseline trained and evaluated
- ANN implementation
- ANN tuning and stabilization
- ANN ensemble prediction for more stable final inference
- fair XGBoost vs ANN comparison on the same split
- dashboard refined for faculty presentation
- detailed final report and final presentation materials prepared

**What to say:**
This slide summarizes what is implemented in the final semester version of the project. The system now includes the complete collected-data workflow, two implemented supervised models, a fair evaluation protocol, and a cleaned presentation layer for final review.

**Image to add:**
- Optional: no image required

---

## Slide 5 - Data Sources
**Title:**  
Collected Data Sources

**Write on slide:**
- PM$_{2.5}$ observations collected from OpenAQ
- Meteorological data collected from ERA5-Land through CDS
- Elevation data collected from terrain source / SRTM-style raster extraction
- Urban context features collected through OSM-based processing
- All collected outputs integrated into one reproducible bundle

**Important wording on slide:**
- Data were collected from APIs and external geospatial sources.
- The final modeling table was created by harmonizing these collected sources.

**What to say:**
The project is based on collected data, not manually created tables. PM$_{2.5}$, meteorology, elevation, and context information were all obtained from external sources and then integrated into one machine-learning-ready structure.

**Image to add:**
- `slide05_data_sources_infographic.png`

---

## Slide 6 - Data Collection Pipeline
**Title:**  
How the Dataset Was Built

**Write on slide as flow:**
1. Define the station framework over Delhi-NCR  
2. Collect PM$_{2.5}$ observations  
3. Collect meteorology  
4. Attach elevation  
5. Attach urban-context variables  
6. Merge all sources into a reproducible bundle  
7. Build a station-date master table

**What to say:**
The pipeline begins with a station framework and then brings in multiple external sources. After collection, each source is standardized and merged so that every record represents a station-date environmental snapshot.

**Image to add:**
- `slide06_collection_pipeline.png`

---

## Slide 7 - Station Framework and Coverage
**Title:**  
Spatial Coverage Used in the Project

**Write on slide:**
- 40 harmonized station IDs across the Delhi region
- Common indexing used for joining PM$_{2.5}$, weather, and static attributes
- Two full years of daily data in the current bundle
- Total PM$_{2.5}$ rows: 29,240
- Total ERA5 rows: 29,240

**What to say:**
The project uses a consistent station framework so that all sources can be merged into one common structure. This is necessary because different source systems do not naturally share the same ready-to-use machine learning schema.

**Image to add:**
- `slide07_station_coverage_map.png`

---

## Slide 8 - Dataset Inventory
**Title:**  
Final Dataset Bundle and Modeling Files

**Write on slide:**
- `stations_urban.csv` -> station coordinates + context fields
- `stations_elevation.csv` -> elevation per station
- `openaq_pm25.csv` -> daily PM$_{2.5}$ target values
- `era5_meteo.csv` -> daily meteorological values
- `delhi_pm25_master.csv` -> final merged modeling table
- `model_predictions_compare.csv` -> final ANN vs XGBoost test predictions
- `model_comparison_metrics.csv` -> final metric summary

**Write key numbers on slide:**
- 40 stations
- 29,240 master rows
- 2023-01-01 to 2024-12-31 data coverage
- 5,880 final test rows

**What to say:**
The output is not just one notebook result. The project produces a complete data and artifact set, including raw collected files, master table, prediction outputs, feature importance, and final comparison metrics.

**Image to add:**
- `slide14_artifacts_snapshot.png`

---

## Slide 9 - Dataset Statistics
**Title:**  
PM$_{2.5}$ Data Description and Statistical Summary

**Write on slide:**
- PM$_{2.5}$ mean: 118.48
- median: 96.99
- standard deviation: 71.75
- minimum: 5.00
- maximum: 308.52
- right-skewed distribution with strong winter pollution episodes
- values are much lower in monsoon months and much higher in winter months

**Also write:**
- highest monthly means: January, November, December
- lowest monthly means: July, August, September

**What to say:**
The distribution itself already tells us that the target is challenging. The spread is large and strongly seasonal. This means the models must learn both broad temporal structure and short-term fluctuation.

**Images to add:**
- `pm25_distribution.png`
- `monthly_mean_pm25.png`

---

## Slide 10 - Feature Engineering
**Title:**  
Features Created for Model Training

**Write on slide in grouped form:**

**Temporal features**
- month, day-of-week, day-of-year
- sine/cosine cyclic encodings

**Historical PM$_{2.5}$ features**
- lag1, lag2, lag3, lag7
- rolling 3-day mean
- rolling 7-day mean
- rolling 7-day standard deviation

**Environmental features**
- 2-meter temperature
- total precipitation

**Spatial/context features**
- latitude, longitude
- elevation
- urban context attributes

**What to say:**
The model is not trained only on raw PM$_{2.5}$. We explicitly engineered temporal, seasonal, meteorological, and lag-memory features so that the learning algorithms can capture recurrence, seasonality, and short-term dependence.

**Image to add:**
- `slide08_preprocessing_pipeline.png`

---

## Slide 11 - Correlation and Data Behavior
**Title:**  
What the Data Tells Us Before Modeling

**Write on slide:**
- PM$_{2.5}$ has strong positive correlation with lag-based features
- Rolling mean features are among the strongest structured signals
- Temperature has a strong negative relationship with PM$_{2.5}$
- Precipitation also shows a negative relationship
- This confirms that seasonality and recent historical memory dominate the problem

**What to say:**
Before training the models, we studied the structure of the data. The correlations already indicate that recent PM$_{2.5}$ history and seasonal effects are major drivers. This helps justify both the selected features and the expected behavior of the models.

**Images to add:**
- `correlation_heatmap.png`
- `missing_values_lags.png`

---

## Slide 12 - XGBoost Model
**Title:**  
XGBoost Baseline Implementation

**Write on slide:**
- Model used: XGBoost Regressor
- Trained on log-transformed PM$_{2.5}$ target
- Uses engineered temporal, meteorological, and lag features
- Uses station identity information through encoded station columns
- Strong baseline for structured tabular environmental data

**Parameter summary on slide:**
- `n_estimators = 1200`
- `max_depth = 6`
- `learning_rate = 0.03`
- subsampling and regularization enabled

**What to say:**
XGBoost is used as the main baseline model because it performs strongly on structured tabular data. It captures nonlinear relationships effectively and provides interpretable feature-importance output.

**Image to add:**
- `slide12_feature_importance_top10.png`

---

## Slide 13 - ANN Model
**Title:**  
ANN Implementation

**Write on slide:**
- Model family: Multilayer Perceptron Regressor
- Hidden layers: `(256, 128)`
- Input scaling using StandardScaler
- Target scaling using transformed target regression
- Early stopping enabled
- Final prediction stabilized through a 5-seed ANN ensemble

**Write on slide:**
- ANN was trained on exactly the same feature set as XGBoost
- ANN used exactly the same test split as XGBoost
- Therefore, the final comparison is fair

**What to say:**
The ANN was tuned and stabilized for final evaluation. The final version uses a stronger architecture and multiple seeds so that the ANN result is not dependent on one unstable run.

**Image to add:**
- Optional: no dedicated image needed, or use a simple architecture schematic you create manually in PPT

---

## Slide 14 - Evaluation Protocol
**Title:**  
How the Models Were Evaluated Fairly

**Write on slide:**
- Chronological split used for each station
- First 80\% of each station timeline used for training
- Last 20\% used for testing
- Same feature space used for both models
- Same test set used for both models
- Same metrics used for both models

**Metrics on slide:**
- MAE
- RMSE
- R$^2$
- Bias
- MAPE / sMAPE

**What to say:**
This slide is important because it establishes fairness. The comparison is not valid unless both models are trained and tested under the same conditions. In this project, the split and metrics were kept identical.

**Image to add:**
- `slide10_chronological_split.png`

---

## Slide 15 - Final ANN vs XGBoost Comparison
**Title:**  
Final Test Results

**Write on slide as table:**

| Model | RMSE | MAE | R$^2$ |
|---|---:|---:|---:|
| ANN | 21.96 | 17.06 | 0.9295 |
| XGBoost | 22.36 | 17.21 | 0.9269 |

**Interpretation text on slide:**
- ANN is slightly better than XGBoost on the final test set
- The performance difference is modest, not extreme
- Both models capture the broader trend well
- ANN gives the better final result under the same protocol

**What to say:**
The most important result is that ANN performs slightly better than XGBoost across the main error metrics. The difference is not dramatic, but it is consistent enough to conclude that ANN is the better model in this experiment.

**Image to add:**
- `model_metrics_comparison.png`

---

## Slide 16 - Station-Level and Feature-Level Insights
**Title:**  
Where the Improvement Comes From

**Write on slide:**
- ANN does not outperform XGBoost by a huge margin, but it improves the final score
- Major predictive drivers are seasonal and lag-based signals
- Station-wise comparison shows ANN improvement on a meaningful subset of stations
- The problem remains difficult for both models when sharp short-term spikes occur

**What to say:**
This result suggests that the current feature space already explains a large part of the behavior. ANN improves over XGBoost, but both models are still mostly driven by seasonality and recent PM$_{2.5}$ history. That is why the final gain is real but limited.

**Images to add:**
- `feature_importance_top10.png`
- `station_mae_gap.png`

---

## Slide 17 - Dashboard and Final Demonstration Layer
**Title:**  
Faculty-Facing Dashboard

**Write on slide:**
- Dashboard refined specifically for faculty evaluation
- Shows only ANN vs XGBoost
- Clean metrics matrix
- Selected-station time-series comparison
- Station ranking table
- Parity plots for both models
- Reduced clutter and improved readability

**What to say:**
The dashboard was redesigned from a debug-style analysis interface into a faculty-facing demonstration view. The purpose is to present only the important comparison results instead of overwhelming the audience with low-value technical clutter.

**Image to add:**
- Optional: add a screenshot of the final Streamlit app after you take one manually

---

## Slide 18 - Conclusion and Final Takeaway
**Title:**  
Conclusion

**Write on slide:**
- Complete project workflow successfully implemented
- Multi-source environmental dataset collected and integrated
- XGBoost baseline built and evaluated
- ANN implemented and tuned for final comparison
- U-Net remains part of the broader project objective and future deep-model direction
- ANN slightly outperforms XGBoost on the final test set
- Work demonstrates a full pipeline from collection to analysis to presentation

**Closing line to speak:**
The final outcome is not just a trained model, but a complete PM$_{2.5}$ estimation pipeline with reproducible data preparation, fair model evaluation, and presentation-ready outputs. Among the implemented models, ANN achieved the best result.

**Image to add:**
- Optional: no image required, or use `slide14_artifacts_snapshot.png` again in smaller form

---

## Optional Backup Slide 1 - Detailed Dataset Schema
**Title:**  
Detailed Dataset Schema

**Write on slide:**
- station-level columns
- meteorological columns
- target column
- lag features
- rolling features
- cyclical time features

**Use when asked:**
- "Explain the attributes in the dataset"
- "What exactly was used as input to the models?"

**Image to add:**
- Optional: no image required

---

## Optional Backup Slide 2 - Why the Difference Between Models Is Small
**Title:**  
Why ANN and XGBoost Are Close

**Write on slide:**
- same dataset
- same features
- same split
- strong temporal and seasonal signal already captured by both
- limited feature richness for sudden spikes
- ANN gives improvement, but the feature representation still constrains both models

**Use when asked:**
- "Why is the gain not large?"
- "If ANN is better, why is it only slightly better?"

**Image to add:**
- Optional: no image required

---

## Images Folder for PPT
Use the following files from `presentation/images/`:

- `slide02_pm25_context_trend.png`
- `slide05_data_sources_infographic.png`
- `slide06_collection_pipeline.png`
- `slide07_station_coverage_map.png`
- `slide08_preprocessing_pipeline.png`
- `slide10_chronological_split.png`
- `slide12_feature_importance_top10.png`
- `slide14_artifacts_snapshot.png`
- `pm25_distribution.png`
- `monthly_mean_pm25.png`
- `correlation_heatmap.png`
- `missing_values_lags.png`
- `model_metrics_comparison.png`
- `feature_importance_top10.png`
- `station_mae_gap.png`

---

## Suggested Delivery Order
- Slides 1-4: 2.5 minutes
- Slides 5-8: 3 minutes
- Slides 9-14: 4.5 minutes
- Slides 15-18: 3 minutes
- Backup slides: if faculty asks for details

This order keeps the presentation balanced between dataset understanding, implementation depth, and final comparative result.

