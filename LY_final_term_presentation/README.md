# Final-Term Report Package

This folder contains the LaTeX source and figures for the final-term report.

## Main file

- `final_term_report.tex`

## Included figures

- `figures/monthly_mean_pm25.png`
- `figures/pm25_distribution.png`
- `figures/correlation_heatmap.png`
- `figures/model_metrics_comparison.png`
- `figures/feature_importance_top10.png`
- `figures/station_mae_gap.png`
- `figures/missing_values_lags.png`

The report also expects a local folder named:

- `images/`

Upload the reused presentation images into that folder in Overleaf. The reused filenames referenced by the report are:

- `slide05_data_sources_infographic.png`
- `slide06_collection_pipeline.png`
- `slide07_station_coverage_map.png`
- `slide08_preprocessing_pipeline.png`
- `slide10_chronological_split.png`
- `slide14_artifacts_snapshot.png`

## Compile

From this folder:

```powershell
pdflatex -interaction=nonstopmode final_term_report.tex
pdflatex -interaction=nonstopmode final_term_report.tex
```

If your LaTeX installation includes `latexmk`, you can also use:

```powershell
latexmk -pdf final_term_report.tex
```

## Notes

- The current machine used to prepare this report does not have `pdflatex` installed, so compilation was not verified locally.
- The report text is based on the current saved dataset and output artifacts in the project workspace.
- The report is intentionally detailed and structured for printed submission / spiral binding.
