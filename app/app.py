import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.inference import (  # noqa: E402
    PredictionBatch,
    PredictionSummary,
    TimeSeriesInput,
    compare_forecasts,
    extract_time_series,
    load_trained_model,
    predict_batch,
    summarize_predictions,
)


@st.cache_resource
def load_meta_model(
    model_path: str,
    columns_path: str,
) -> tuple[RandomForestClassifier, list[str]]:
    """Load the trained classifier and its feature schema once per app process."""
    return load_trained_model(model_path, columns_path)


def format_number(value: float) -> str:
    return f"{value:,.4g}"


def format_confidence(value: float | str) -> str:
    if isinstance(value, str) and value.endswith("%"):
        return value
    return f"{float(value):.1%}"


def series_label(index: int, series: TimeSeriesInput) -> str:
    return f"{series.series_id} (series {index + 1})"


st.set_page_config(page_title="Time series model recommendations", layout="wide")
st.title("Meta-learning time series forecasting")
st.caption(
    "Upload a wide time-series CSV to get a forecasting model recommendation "
    "for every series."
)

st.header("Upload dataset")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=(
        "Each row is one time series. The first non-numeric column identifies "
        "the series; numeric columns contain sequential observations."
    ),
)

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

file_bytes = uploaded_file.getvalue()
upload_digest = hashlib.sha256(file_bytes).hexdigest()
try:
    dataframe = pd.read_csv(uploaded_file)
except (
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    UnicodeDecodeError,
    ValueError,
) as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

st.header("Dataset preview")
st.dataframe(
    dataframe.head(5),
    hide_index=True,
    alt="First five rows of the uploaded dataset",
)

try:
    series_inputs = extract_time_series(dataframe)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

model_path = root / "models" / "random_forest.pkl"
columns_path = root / "models" / "feature_columns.json"
try:
    model, feature_columns = load_meta_model(
        str(model_path),
        str(columns_path),
    )
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

if st.session_state.get("upload_digest") != upload_digest:
    started_at = time.perf_counter()
    progress_bar = st.progress(0.0, text="Preparing time series...")
    with st.status("Analyzing uploaded series...", expanded=True) as status:
        def update_progress(completed: int, total: int) -> None:
            status.update(label=f"Analyzing series {completed} of {total}")
            progress_bar.progress(completed / total)

        batch: PredictionBatch = predict_batch(
            series_inputs,
            model,
            feature_columns,
            progress_callback=update_progress,
        )
        processing_seconds = time.perf_counter() - started_at
        status.update(
            label=(
                "Analysis complete: "
                f"{len(batch.recommendations)} recommendations generated"
            ),
            state="complete",
            expanded=False,
        )
    progress_bar.empty()
    st.session_state["upload_digest"] = upload_digest
    st.session_state["batch_results"] = batch.recommendations
    st.session_state["batch_details"] = batch.details
    st.session_state["batch_errors"] = batch.errors
    st.session_state["batch_processing_seconds"] = processing_seconds
    st.session_state.pop("forecast_result_key", None)

results = st.session_state["batch_results"]
detail_by_index = st.session_state["batch_details"]
row_errors = st.session_state["batch_errors"]
processing_seconds = st.session_state["batch_processing_seconds"]

raw_observations = sum(len(item.values) for item in series_inputs)
missing_values = sum(item.removed_count for item in series_inputs)
st.header("Dataset summary")
dataset_metrics = st.columns(4)
dataset_metrics[0].metric("Total time series", f"{len(series_inputs):,}")
dataset_metrics[1].metric("Observations", f"{raw_observations:,}")
dataset_metrics[2].metric("Missing values", f"{missing_values:,}")
dataset_metrics[3].metric("Processing time", f"{processing_seconds:.2f} sec")

if results.empty:
    st.error("No series had valid numeric observations for a recommendation.")
    with st.expander(f"Skipped series ({len(row_errors)})"):
        for index, message in row_errors.items():
            st.write(f"**{series_label(index, series_inputs[index])}:** {message}")
    st.stop()

prediction_summary: PredictionSummary = summarize_predictions(results)
st.header("Prediction summary")
prediction_metrics = st.columns(5)
prediction_metrics[0].metric(
    "Predictions generated",
    f"{prediction_summary.total:,}",
)
prediction_metrics[1].metric(
    "ARIMA recommendations",
    f"{prediction_summary.arima_count:,}",
)
prediction_metrics[2].metric(
    "Prophet recommendations",
    f"{prediction_summary.prophet_count:,}",
)
prediction_metrics[3].metric(
    "Average confidence",
    f"{prediction_summary.average_confidence:.1%}",
)
prediction_metrics[4].metric(
    "Overall recommended model",
    prediction_summary.overall_model,
)
st.caption(prediction_summary.explanation)
if row_errors:
    st.warning(
        f"{len(row_errors)} of {len(series_inputs)} series were skipped due "
        "to missing observations or invalid features."
    )
    with st.expander("Skipped series details"):
        for index, message in row_errors.items():
            st.write(f"**{series_label(index, series_inputs[index])}:** {message}")

st.header("Recommendation table")
display_results = results.copy()
display_results["Confidence"] = display_results["Confidence"].map(
    format_confidence
)
st.dataframe(
    display_results,
    hide_index=True,
    alt="Model recommendations and confidence for each time series",
)
st.download_button(
    "Download recommendation table",
    data=display_results.to_csv(index=False).encode("utf-8"),
    file_name="forecast_recommendations.csv",
    mime="text/csv",
)

st.header("Series inspection")
selected_index = st.selectbox(
    "Select a predicted series",
    options=list(detail_by_index),
    format_func=lambda index: series_label(index, series_inputs[index]),
)
selected_series = series_inputs[selected_index]
selected_detail = detail_by_index[selected_index]
selected_features = selected_detail.features
probabilities = selected_detail.probabilities

st.subheader("Meta features")
feature_metrics = st.columns(5)
for feature_number, feature_name in enumerate(feature_columns):
    feature_metrics[feature_number % len(feature_metrics)].metric(
        feature_name.replace("_", " ").title(),
        format_number(float(selected_features[feature_name])),
    )

st.subheader("Prediction details")
prediction_details = st.columns(4)
prediction_details[0].metric(
    "Recommended model",
    selected_detail.recommendation,
)
prediction_details[1].metric("Confidence", f"{selected_detail.confidence:.1%}")
prediction_details[2].metric(
    "Probability of ARIMA",
    f"{float(probabilities['ARIMA']):.1%}",
)
prediction_details[3].metric(
    "Probability of Prophet",
    f"{float(probabilities['Prophet']):.1%}",
)

comparison_key = (upload_digest, selected_index)
if st.session_state.get("forecast_result_key") == comparison_key:
    scores = st.session_state["forecast_scores"]
    forecasts = st.session_state["forecast_plot"]
    best_actual_model = min(scores, key=scores.get)
    st.header("Forecast validation")
    comparison_metrics = st.columns(2)
    comparison_metrics[0].metric("ARIMA RMSE", format_number(scores["ARIMA"]))
    comparison_metrics[1].metric(
        "Prophet RMSE",
        format_number(scores["Prophet"]),
    )
    st.metric("Actual best model", best_actual_model)
    if selected_detail.recommendation == best_actual_model:
        st.success("The Random Forest recommendation matched the lower-RMSE model.")
    else:
        st.warning(
            f"The Random Forest recommended {selected_detail.recommendation}; "
            f"{best_actual_model} had the lower holdout RMSE."
        )
    st.subheader("Forecast plot")
    st.line_chart(
        forecasts,
        alt="Actual holdout values and ARIMA and Prophet forecasts",
    )

st.header("Compare forecasts")
if st.button("Compare Forecasts", type="primary"):
    with st.spinner("Training ARIMA and Prophet for the selected series..."):
        try:
            scores, forecasts = compare_forecasts(selected_series.values)
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            OverflowError,
            OSError,
        ) as exc:
            st.error(f"Forecast comparison failed: {exc}")
        else:
            st.session_state["forecast_result_key"] = comparison_key
            st.session_state["forecast_scores"] = scores
            st.session_state["forecast_plot"] = forecasts
            st.rerun()
