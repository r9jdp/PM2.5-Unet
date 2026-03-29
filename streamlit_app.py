from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="PM2.5 XGBoost Dashboard", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_outputs"
BUNDLE_DIR = BASE_DIR / "pm25_delhi_bundle"

PRED_PATH = OUTPUT_DIR / "predictions_all.csv"
FI_PATH = OUTPUT_DIR / "feature_importance.csv"
MASTER_PATH = OUTPUT_DIR / "delhi_pm25_master.csv"
OPENAQ_PATH = BUNDLE_DIR / "openaq_pm25.csv"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def metric_block(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - actual
    abs_err = err.abs()
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err.pow(2)).mean()))

    ss_res = float(((actual - pred) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    non_zero = actual != 0
    mape = float((abs_err[non_zero] / actual[non_zero].abs()).mean() * 100) if non_zero.any() else float("nan")
    smape = float((2 * abs_err / (actual.abs() + pred.abs()).replace(0, np.nan)).mean() * 100)
    bias = float(err.mean())

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "sMAPE": smape,
        "Bias": bias,
    }


def build_persistence_baseline(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy().sort_values(["station_id", "date"]).reset_index(drop=True)
    out["pred_persistence"] = out.groupby("station_id")["actual_pm25"].shift(1)
    return out.dropna(subset=["pred_persistence"]) 


def header() -> None:
    st.title("PM2.5 Prediction Dashboard")
    st.caption("XGBoost performance summary and diagnostics for professor presentation")


header()

st.sidebar.header("Data Sources")
st.sidebar.write(f"Predictions: {PRED_PATH.name}")
st.sidebar.write(f"Feature Importance: {FI_PATH.name}")
st.sidebar.write(f"Training Master: {MASTER_PATH.name}")
st.sidebar.write(f"OpenAQ Target File: {OPENAQ_PATH.name}")

missing = [p for p in [PRED_PATH, FI_PATH, MASTER_PATH] if not p.exists()]
if missing:
    st.error("Missing required model output files. Run the training notebook first.")
    st.stop()

pred_df = load_csv(PRED_PATH)
fi_df = load_csv(FI_PATH)
master_df = load_csv(MASTER_PATH)

if pred_df.empty:
    st.error("predictions_all.csv is empty.")
    st.stop()

pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
pred_df = pred_df.dropna(subset=["date", "actual_pm25", "pred_pm25"]).copy()
pred_df["actual_pm25"] = pd.to_numeric(pred_df["actual_pm25"], errors="coerce")
pred_df["pred_pm25"] = pd.to_numeric(pred_df["pred_pm25"], errors="coerce")
pred_df = pred_df.dropna(subset=["actual_pm25", "pred_pm25"]).copy()

station_options = sorted(pred_df["station_id"].astype(str).unique())
selected_stations = st.sidebar.multiselect(
    "Filter station(s)",
    options=station_options,
    default=station_options,
)

if not selected_stations:
    st.warning("Select at least one station.")
    st.stop()

view_df = pred_df[pred_df["station_id"].astype(str).isin(selected_stations)].copy()
if view_df.empty:
    st.warning("No rows after station filter.")
    st.stop()

metrics = metric_block(view_df["actual_pm25"], view_df["pred_pm25"])

# Optional baseline from persistence lag-1 on same prediction table.
baseline_df = build_persistence_baseline(view_df)
baseline_metrics = None
if not baseline_df.empty:
    baseline_metrics = metric_block(baseline_df["actual_pm25"], baseline_df["pred_persistence"])

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Rows", f"{len(view_df):,}")
col2.metric("Stations", f"{view_df['station_id'].nunique()}")
col3.metric("MAE", f"{metrics['MAE']:.2f}")
col4.metric("RMSE", f"{metrics['RMSE']:.2f}")
col5.metric("R²", f"{metrics['R2']:.3f}")
col6.metric("sMAPE %", f"{metrics['sMAPE']:.2f}")

st.markdown("### Metrics Matrix")
rows = [
    {
        "Model": "XGBoost Final",
        "MAE (ug/m3)": metrics["MAE"],
        "RMSE (ug/m3)": metrics["RMSE"],
        "R2": metrics["R2"],
        "MAPE (%)": metrics["MAPE"],
        "sMAPE (%)": metrics["sMAPE"],
        "Bias (Pred-Actual)": metrics["Bias"],
    }
]
if baseline_metrics is not None:
    rows.append(
        {
            "Model": "Persistence (lag-1)",
            "MAE (ug/m3)": baseline_metrics["MAE"],
            "RMSE (ug/m3)": baseline_metrics["RMSE"],
            "R2": baseline_metrics["R2"],
            "MAPE (%)": baseline_metrics["MAPE"],
            "sMAPE (%)": baseline_metrics["sMAPE"],
            "Bias (Pred-Actual)": baseline_metrics["Bias"],
        }
    )

metrics_df = pd.DataFrame(rows)
st.dataframe(
    metrics_df.style.format(
        {
            "MAE (ug/m3)": "{:.2f}",
            "RMSE (ug/m3)": "{:.2f}",
            "R2": "{:.3f}",
            "MAPE (%)": "{:.2f}",
            "sMAPE (%)": "{:.2f}",
            "Bias (Pred-Actual)": "{:.2f}",
        }
    ),
    use_container_width=True,
)

chart_df = view_df.sort_values("date").copy()
chart_df["abs_error"] = (chart_df["actual_pm25"] - chart_df["pred_pm25"]).abs()
chart_df["residual"] = chart_df["pred_pm25"] - chart_df["actual_pm25"]

st.markdown("### Actual vs Predicted Over Time")
line_plot_df = chart_df[["date", "actual_pm25", "pred_pm25"]].set_index("date")
st.line_chart(line_plot_df, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Parity Check")
    parity = chart_df[["actual_pm25", "pred_pm25"]].copy()
    st.scatter_chart(parity, x="actual_pm25", y="pred_pm25", use_container_width=True)
with c2:
    st.markdown("### Residual vs Actual")
    res_plot = chart_df[["actual_pm25", "residual"]].copy()
    st.scatter_chart(res_plot, x="actual_pm25", y="residual", use_container_width=True)

st.markdown("### Worst Error Days")
worst = chart_df.sort_values("abs_error", ascending=False).head(25)
st.dataframe(
    worst[["station_id", "date", "actual_pm25", "pred_pm25", "abs_error", "residual"]],
    use_container_width=True,
)

st.markdown("### Feature Importance")
if not fi_df.empty and {"feature", "importance"}.issubset(fi_df.columns):
    fi_top = fi_df.sort_values("importance", ascending=False).head(20).set_index("feature")
    st.bar_chart(fi_top["importance"], use_container_width=True)
else:
    st.info("feature_importance.csv not available or missing columns.")

st.markdown("### Download Tables")
out_pred = chart_df.to_csv(index=False).encode("utf-8")
out_metrics = metrics_df.to_csv(index=False).encode("utf-8")

b1, b2 = st.columns(2)
with b1:
    st.download_button(
        label="Download Filtered Predictions CSV",
        data=out_pred,
        file_name="dashboard_predictions_filtered.csv",
        mime="text/csv",
    )
with b2:
    st.download_button(
        label="Download Metrics Matrix CSV",
        data=out_metrics,
        file_name="dashboard_metrics_matrix.csv",
        mime="text/csv",
    )

with st.expander("Method Notes For Presentation"):
    st.write(
        "This dashboard summarizes the current XGBoost run using your saved model outputs. "
        "Metrics are displayed in concentration units (ug/m3). For percent-style interpretation, refer to MAPE/sMAPE."
    )
    st.write(
        "When station coverage is limited, results represent temporal predictive performance for available stations, "
        "not city-wide spatial generalization."
    )
