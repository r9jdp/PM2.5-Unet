## Project Structure

This workspace is now split into `core project files` and `parked extras`.

### Core project files

These are the files/folders you should care about for the PM2.5 project:

- `01_build_download_dataset.ipynb`
  Builds and packages the Delhi dataset bundle.
- `02_train_xgboost_model.ipynb`
  Main training notebook. Also contains the ANN comparison experiment.
- `streamlit_app.py`
  Dashboard app for presentation/demo.
- `pm25_delhi_bundle/`
  Main dataset bundle used by the notebook and app.
- `model_outputs/`
  Current saved prediction outputs, feature importance, and master table.
- `LY_mid_term_presentation/`
  Mid-term slides, images, and report materials.
- `LY_final_term_presentation/`
  Use this for final-term presentation files.
- `REPORT.md`
  Short technical summary of the current model workflow.
- `fix_dataset_temp.py`
  Utility for repairing/rebuilding `openaq_pm25.csv` if needed.
- `requirements.txt`
  Python dependencies.

### Parked extras

These were cluttering the root and are not part of the active PM2.5 workflow:

- `unused_or_external/embedding_model_bge_small_en_v1_5/`
  Downloaded sentence-transformer / BGE embedding model files.
  These are unrelated to the PM2.5 training notebook and dashboard.
- `unused_or_external/training_logs/catboost_info/`
  Training log artifacts. Not used by the current PM2.5 pipeline.

### What to open for normal work

If you are working on the project, usually open only:

- `02_train_xgboost_model.ipynb`
- `streamlit_app.py`
- `pm25_delhi_bundle/`
- `model_outputs/`
- `LY_final_term_presentation/`
