from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None


st.set_page_config(
    page_title="PM2.5 Final-Term Comparison",
    page_icon="AQ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_outputs"
PRED_PATH = OUTPUT_DIR / "model_predictions_compare.csv"
METRICS_PATH = OUTPUT_DIR / "model_comparison_metrics.csv"

MODEL_LABELS = {
    "pred_xgb_model": "XGBoost",
    "pred_ann_model": "ANN",
}


@st.cache_data(show_spinner=False)
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_csv(PRED_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    return pred_df, metrics_df


def metric_block(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - actual
    abs_err = err.abs()
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err.pow(2)).mean()))

    ss_res = float(((actual - pred) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    bias = float(err.mean())

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Bias": bias,
    }


def build_metrics_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = pred_df["actual_pm25"]
    for col, label in MODEL_LABELS.items():
        metrics = metric_block(actual, pred_df[col])
        rows.append(
            {
                "Model": label,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def build_station_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, station_df in pred_df.groupby("station_id", sort=True):
        row = {"station_id": station_id, "rows": len(station_df)}
        for col, label in MODEL_LABELS.items():
            metrics = metric_block(station_df["actual_pm25"], station_df[col])
            row[f"{label}_MAE"] = metrics["MAE"]
            row[f"{label}_RMSE"] = metrics["RMSE"]
        row["Better Model"] = "ANN" if row["ANN_MAE"] < row["XGBoost_MAE"] else "XGBoost"
        row["MAE Gap"] = row["XGBoost_MAE"] - row["ANN_MAE"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE Gap", ascending=False).reset_index(drop=True)


def kpi_card(title: str, value: str, tone: str = "neutral") -> str:
    tone_map = {
        "neutral": "#f4efe3",
        "good": "#d8f0d2",
        "warn": "#f4d7c8",
    }
    bg = tone_map.get(tone, tone_map["neutral"])
    return f"""
    <div style="
        background:{bg};
        border-radius:18px;
        padding:18px 20px;
        min-height:108px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    ">
        <div style="font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.72;">{title}</div>
        <div style="font-size:2rem; font-weight:700; margin-top:10px; color:#182218;">{value}</div>
    </div>
    """


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(220,232,214,0.9), transparent 35%),
            linear-gradient(180deg, #f5efe6 0%, #ede4d7 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3 {
        color: #17221a;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.66);
        border: 1px solid rgba(23,34,26,0.08);
        border-radius: 16px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.72);
        border-radius: 16px;
        padding: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not PRED_PATH.exists() or not METRICS_PATH.exists():
    st.error("Missing comparison files. Run `python build_model_comparison.py` first.")
    st.stop()

pred_df, _saved_metrics_df = load_inputs()
pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
pred_df["actual_pm25"] = pd.to_numeric(pred_df["actual_pm25"], errors="coerce")
for col in MODEL_LABELS:
    pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
pred_df = pred_df.dropna(subset=["date", "station_id", "actual_pm25", "pred_xgb_model", "pred_ann_model"]).copy()

overall_metrics = build_metrics_table(pred_df)
station_metrics = build_station_metrics(pred_df)

winner = overall_metrics.iloc[0]
runner_up = overall_metrics.iloc[1]
mae_margin = runner_up["MAE"] - winner["MAE"]
ann_row = overall_metrics[overall_metrics["Model"] == "ANN"].iloc[0]
xgb_row = overall_metrics[overall_metrics["Model"] == "XGBoost"].iloc[0]

st.title("PM2.5 Final-Term Model Comparison")
st.caption("Faculty presentation view: only the two final models, the same split, and the most useful visuals.")

top_left, top_mid, top_right = st.columns([1.35, 1.1, 1.1])
with top_left:
    st.markdown(
        """
        ### Final takeaway
        The dashboard compares only **XGBoost** and **ANN** on the same chronological holdout.
        No auxiliary baselines, no debug rows, and no overplotted multi-station charts.
        """
    )
with top_mid:
    st.markdown(
        kpi_card("Best Model", winner["Model"], "good" if winner["Model"] == "ANN" else "warn"),
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        kpi_card("MAE Margin", f"{mae_margin:.2f} ug/m3", "good"),
        unsafe_allow_html=True,
    )

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("ANN MAE", f"{ann_row['MAE']:.2f}", "good"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("XGBoost MAE", f"{xgb_row['MAE']:.2f}", "neutral"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("ANN R2", f"{ann_row['R2']:.3f}", "good"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Stations", f"{pred_df['station_id'].nunique()}"), unsafe_allow_html=True)

st.markdown("### Overall Metrics")
display_metrics = overall_metrics.copy()
st.dataframe(
    display_metrics.round({"MAE": 2, "RMSE": 2, "R2": 3}),
    use_container_width=True,
    hide_index=True,
)

control_a, control_b = st.columns([1, 1.2])
station_options = sorted(pred_df["station_id"].astype(str).unique())
selected_station = control_a.selectbox("Station", station_options, index=0)
focus_metric = control_b.radio("Station Ranking Metric", ["MAE Gap", "ANN_MAE", "XGBoost_MAE"], horizontal=True)

station_view_df = pred_df[pred_df["station_id"].astype(str) == selected_station].sort_values("date").copy()
date_min = station_view_df["date"].min().date()
date_max = station_view_df["date"].max().date()
default_start = max(date_min, (station_view_df["date"].max() - pd.Timedelta(days=27)).date())
date_window = st.slider(
    "Date window",
    min_value=date_min,
    max_value=date_max,
    value=(default_start, date_max),
)
chart_source = station_view_df[
    (station_view_df["date"].dt.date >= date_window[0]) & (station_view_df["date"].dt.date <= date_window[1])
].copy()
chart_title = f"{selected_station}: daily PM2.5 comparison"

if chart_source.empty:
    st.warning("No rows found for the selected date window.")
    st.stop()

line_df = chart_source.rename(
    columns={
        "actual_pm25": "Actual",
        "pred_xgb_model": "XGBoost",
        "pred_ann_model": "ANN",
    }
)

st.markdown("### Time-Series Comparison")
if alt is not None:
    long_df = line_df.melt("date", var_name="series", value_name="pm25")
    color_scale = alt.Scale(domain=["Actual", "ANN", "XGBoost"], range=["#204a87", "#0b8f55", "#a8512b"])
    chart = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.6)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("pm25:Q", title="PM2.5 (ug/m3)"),
            color=alt.Color("series:N", title="", scale=color_scale),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("pm25:Q", title="PM2.5", format=".2f"),
            ],
        )
        .properties(height=360, title=chart_title)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.line_chart(line_df.set_index("date")[["Actual", "ANN", "XGBoost"]], use_container_width=True)

left, right = st.columns([1.1, 0.9])
with left:
    st.markdown("### Station-Wise Performance")
    station_table = station_metrics.sort_values(focus_metric, ascending=(focus_metric != "MAE Gap")).copy()
    st.dataframe(
        station_table.round(
            {
                "ANN_MAE": 2,
                "XGBoost_MAE": 2,
                "ANN_RMSE": 2,
                "XGBoost_RMSE": 2,
                "MAE Gap": 2,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown("### Faculty Summary")
    better_count = int((station_metrics["Better Model"] == "ANN").sum())
    worse_count = int((station_metrics["Better Model"] == "XGBoost").sum())
    st.markdown(
        f"""
        - **ANN overall MAE:** `{ann_row['MAE']:.2f}`
        - **XGBoost overall MAE:** `{xgb_row['MAE']:.2f}`
        - **ANN better stations:** `{better_count}`
        - **XGBoost better stations:** `{worse_count}`

        **Interpretation**

        The comparison is now presentation-focused:
        only two models, one clean station selector, and one fair metric table.
        """
    )

diag_left, diag_right = st.columns(2)
with diag_left:
    st.markdown("### Parity: XGBoost")
    parity_xgb = chart_source[["actual_pm25", "pred_xgb_model"]].rename(columns={"pred_xgb_model": "pred_pm25"})
    if alt is not None:
        parity_chart = (
            alt.Chart(parity_xgb)
            .mark_circle(size=72, opacity=0.75, color="#a8512b")
            .encode(
                x=alt.X("actual_pm25:Q", title="Actual PM2.5"),
                y=alt.Y("pred_pm25:Q", title="Predicted PM2.5"),
                tooltip=[alt.Tooltip("actual_pm25:Q", format=".2f"), alt.Tooltip("pred_pm25:Q", format=".2f")],
            )
            .properties(height=280)
        )
        st.altair_chart(parity_chart, use_container_width=True)
    else:
        st.scatter_chart(parity_xgb, x="actual_pm25", y="pred_pm25", use_container_width=True)

with diag_right:
    st.markdown("### Parity: ANN")
    parity_ann = chart_source[["actual_pm25", "pred_ann_model"]].rename(columns={"pred_ann_model": "pred_pm25"})
    if alt is not None:
        parity_chart = (
            alt.Chart(parity_ann)
            .mark_circle(size=72, opacity=0.75, color="#0b8f55")
            .encode(
                x=alt.X("actual_pm25:Q", title="Actual PM2.5"),
                y=alt.Y("pred_pm25:Q", title="Predicted PM2.5"),
                tooltip=[alt.Tooltip("actual_pm25:Q", format=".2f"), alt.Tooltip("pred_pm25:Q", format=".2f")],
            )
            .properties(height=280)
        )
        st.altair_chart(parity_chart, use_container_width=True)
    else:
        st.scatter_chart(parity_ann, x="actual_pm25", y="pred_pm25", use_container_width=True)
