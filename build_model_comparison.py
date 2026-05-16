from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_outputs"
MASTER_PATH = OUTPUT_DIR / "delhi_pm25_master.csv"
COMPARE_PRED_PATH = OUTPUT_DIR / "model_predictions_compare.csv"
COMPARE_METRICS_PATH = OUTPUT_DIR / "model_comparison_metrics.csv"

ANN_ENSEMBLE_SEEDS = [11, 22, 33, 44, 55]


def metric_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_pred - y_true
    abs_error = np.abs(residual)
    non_zero = y_true != 0
    row = {
        "model": model_name,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(residual)),
        "smape": float(np.nanmean(2 * abs_error / np.where((np.abs(y_true) + np.abs(y_pred)) == 0, np.nan, np.abs(y_true) + np.abs(y_pred))) * 100),
        "mape": float(np.nanmean(np.where(non_zero, abs_error / np.abs(y_true), np.nan)) * 100),
        "n_test_rows": int(y_true.shape[0]),
    }
    return row


def build_ann_model(seed: int) -> Pipeline:
    ann_regressor = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=5e-4,
        batch_size=128,
        max_iter=1200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("mlp", TransformedTargetRegressor(regressor=ann_regressor, transformer=StandardScaler())),
        ]
    )


def build_outputs() -> None:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {MASTER_PATH}")

    base = pd.read_csv(MASTER_PATH)
    base["station_id"] = base["station_id"].astype(str)
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["station_id", "date", "pm25", "log_pm25"]).copy()
    base = base.sort_values(["station_id", "date"]).reset_index(drop=True)

    numeric_cols = [c for c in base.columns if c not in {"station_id", "date"}]
    for col in numeric_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    g = base.groupby("station_id", group_keys=False)
    base["rank_station"] = g.cumcount() + 1
    base["n_station"] = g["pm25"].transform("size")
    base["is_test"] = base["rank_station"] > (0.8 * base["n_station"])

    train_raw = base[~base["is_test"]].copy()
    test_raw = base[base["is_test"]].copy()

    clim_sm = train_raw.groupby(["station_id", "month"])["pm25"].mean().rename("clim_station_month").reset_index()
    clim_s = train_raw.groupby("station_id")["pm25"].mean().rename("clim_station").reset_index()
    global_mean = float(train_raw["pm25"].mean())

    train_raw = train_raw.merge(clim_sm, on=["station_id", "month"], how="left")
    train_raw = train_raw.merge(clim_s, on="station_id", how="left")
    train_raw["clim_station_month"] = (
        train_raw["clim_station_month"].fillna(train_raw["clim_station"]).fillna(global_mean)
    )

    test_raw = test_raw.merge(clim_sm, on=["station_id", "month"], how="left")
    test_raw = test_raw.merge(clim_s, on="station_id", how="left")
    test_raw["clim_station_month"] = (
        test_raw["clim_station_month"].fillna(test_raw["clim_station"]).fillna(global_mean)
    )

    train_df = pd.get_dummies(train_raw.copy(), columns=["station_id"], prefix="sid")
    test_df = pd.get_dummies(test_raw.copy(), columns=["station_id"], prefix="sid")
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    candidate_features = [
        "lat",
        "lon",
        "building_density",
        "road_density",
        "elevation",
        "temp_2m",
        "total_precip",
        "wind_u",
        "wind_v",
        "wind_speed",
        "surface_pressure",
        "dewpoint_2m",
        "month",
        "dayofweek",
        "dayofyear",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "pm25_lag1",
        "pm25_lag2",
        "pm25_lag3",
        "pm25_lag7",
        "pm25_roll3_mean",
        "pm25_roll7_mean",
        "pm25_roll7_std",
        "clim_station_month",
    ]
    station_dummy_cols = [c for c in train_df.columns if c.startswith("sid_")]
    features = [c for c in candidate_features if c in train_df.columns] + station_dummy_cols

    train_df = train_df.dropna(subset=features + ["log_pm25"]).copy()
    test_df = test_df.dropna(subset=features + ["log_pm25"]).copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test empty after filtering. Check delhi_pm25_master.csv")

    xgb_model = XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
    )
    xgb_model.fit(train_df[features], train_df["log_pm25"], verbose=False)

    test_eval = test_raw.loc[test_df.index].copy()
    actual = np.expm1(test_df["log_pm25"].to_numpy())
    pred_xgb_model = np.clip(np.expm1(xgb_model.predict(test_df[features])), 1, 500)
    ann_predictions = []
    for seed in ANN_ENSEMBLE_SEEDS:
        ann_model = build_ann_model(seed)
        ann_model.fit(train_df[features], train_df["log_pm25"])
        ann_predictions.append(np.clip(np.expm1(ann_model.predict(test_df[features])), 1, 500))
    pred_ann_model = np.mean(ann_predictions, axis=0)
    compare_pred_df = test_eval[["station_id", "date"]].copy()
    compare_pred_df["actual_pm25"] = actual
    compare_pred_df["split"] = "test"
    compare_pred_df["pred_xgb_model"] = pred_xgb_model
    compare_pred_df["pred_ann_model"] = pred_ann_model
    compare_pred_df["pred_xgboost"] = pred_xgb_model
    compare_pred_df["pred_ann"] = pred_ann_model
    compare_pred_df["residual_xgboost"] = pred_xgb_model - actual
    compare_pred_df["residual_ann"] = pred_ann_model - actual
    compare_pred_df["abs_error_xgboost"] = np.abs(compare_pred_df["residual_xgboost"])
    compare_pred_df["abs_error_ann"] = np.abs(compare_pred_df["residual_ann"])
    compare_pred_df = compare_pred_df.sort_values(["station_id", "date"]).reset_index(drop=True)

    metrics_df = pd.DataFrame(
        [
            metric_row("ANN", actual, pred_ann_model),
            metric_row("XGBoost", actual, pred_xgb_model),
        ]
    ).sort_values(["mae", "rmse"]).reset_index(drop=True)

    compare_pred_df.to_csv(COMPARE_PRED_PATH, index=False)
    metrics_df.to_csv(COMPARE_METRICS_PATH, index=False)

    print("Saved comparison artifacts:")
    print(COMPARE_PRED_PATH)
    print(COMPARE_METRICS_PATH)
    print("ANN ensemble seeds:", ANN_ENSEMBLE_SEEDS)
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    build_outputs()
