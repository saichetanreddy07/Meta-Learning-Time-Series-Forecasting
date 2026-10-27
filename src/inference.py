from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.preprocessing import preprocess_series, train_test_split
from src.evaluation.metrics import rmse
from src.features.feature_extraction import extract_meta_features
from src.models.arima_model import forecast_arima, train_arima
from src.models.prophet_model import forecast_prophet, train_prophet


@dataclass(frozen=True)
class TimeSeriesInput:
    """One row of uploaded observations and its identifier."""

    series_id: str
    values: pd.Series
    removed_count: int


@dataclass(frozen=True)
class SeriesPrediction:
    """Classifier output and the meta-features used for one series."""

    features: dict[str, float | int]
    recommendation: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class PredictionBatch:
    """Recommendations, row-level details, and errors for one uploaded file."""

    recommendations: pd.DataFrame
    details: dict[int, SeriesPrediction]
    errors: dict[int, str]


@dataclass(frozen=True)
class PredictionSummary:
    """Aggregated recommendation counts and confidence for an uploaded file."""

    total: int
    arima_count: int
    prophet_count: int
    average_confidence: float
    overall_model: str
    explanation: str


def extract_time_series(frame: pd.DataFrame) -> list[TimeSeriesInput]:
    """Read each row as a series, using the first non-numeric column as its ID."""
    if frame.empty:
        raise ValueError("The CSV has no time series.")
    if frame.shape[1] < 2:
        raise ValueError(
            "The CSV must contain a series ID column and numeric observation columns."
        )

    non_numeric_indices = []
    numeric_columns = {}
    for column_index, column in enumerate(frame.columns):
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid_values = frame[column].notna() & converted.isna()
        if invalid_values.any():
            non_numeric_indices.append(column_index)
        else:
            numeric_columns[column_index] = converted

    if not non_numeric_indices:
        raise ValueError(
            "The CSV must include a non-numeric series ID column and numeric "
            "observation columns."
        )
    if len(non_numeric_indices) > 1:
        extra_columns = [
            str(frame.columns[index]) for index in non_numeric_indices[1:]
        ]
        raise ValueError(
            "Observation columns contain non-numeric values: "
            f"{', '.join(extra_columns)}."
        )

    identifier_index = non_numeric_indices[0]
    observation_indices = [
        index for index in range(frame.shape[1]) if index != identifier_index
    ]

    if not any(
        np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).any()
        for values in numeric_columns.values()
    ):
        raise ValueError("No numeric observations were found in the CSV.")

    series_inputs = []
    for row_index in range(len(frame)):
        identifier = frame.iat[row_index, identifier_index]
        if pd.isna(identifier) or not str(identifier).strip():
            series_id = f"Series {row_index + 1}"
        else:
            series_id = str(identifier).strip()

        row_values = np.asarray(
            [
                numeric_columns[col_idx].iat[row_index]
                for col_idx in observation_indices
            ],
            dtype=float,
        )
        valid_mask = np.isfinite(row_values)
        values = pd.Series(row_values[valid_mask], name=series_id, dtype=float)
        series_inputs.append(
            TimeSeriesInput(
                series_id=series_id,
                values=values,
                removed_count=len(row_values) - int(valid_mask.sum()),
            )
        )

    return series_inputs


def load_trained_model(
    model_path: str | Path,
    columns_path: str | Path,
) -> tuple[RandomForestClassifier, list[str]]:
    """Load and validate the saved classifier and its feature schema."""
    model_file = Path(model_path)
    columns_file = Path(columns_path)
    if not model_file.is_file() or not columns_file.is_file():
        raise FileNotFoundError(
            "The trained recommendation model is missing. Run "
            "`python src/pipeline/run_pipeline.py` to create it."
        )

    model = joblib.load(model_file)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("The saved recommendation model is not a Random Forest.")

    with columns_file.open("r", encoding="utf-8") as file:
        feature_columns = json.load(file)
    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or not all(isinstance(column, str) for column in feature_columns)
        or len(set(feature_columns)) != len(feature_columns)
    ):
        raise ValueError("The saved model feature metadata is invalid.")
    if (
        getattr(model, "feature_names_in_", None) is not None
        and list(model.feature_names_in_) != feature_columns
    ):
        raise ValueError("The saved model and feature metadata do not match.")
    if getattr(model, "n_features_in_", len(feature_columns)) != len(
        feature_columns
    ):
        raise ValueError("The saved model feature count does not match its metadata.")
    if set(model.classes_) != {"ARIMA", "Prophet"}:
        raise ValueError("The saved classifier must contain ARIMA and Prophet.")

    return model, feature_columns


def _extract_series_features(
    values: pd.Series,
    feature_columns: list[str],
) -> dict[str, float | int]:
    """Extract meta-features and validate them against the model schema."""
    features = extract_meta_features(values)
    missing = sorted(set(feature_columns).difference(features))
    if missing:
        raise ValueError(f"Missing classifier features: {', '.join(missing)}.")
    if not all(np.isfinite(features[col]) for col in feature_columns):
        raise ValueError("Meta-features contain non-finite values.")
    return features


def predict_batch(
    series_inputs: list[TimeSeriesInput],
    model: RandomForestClassifier,
    feature_columns: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> PredictionBatch:
    """Extract row features and predict all eligible series in one model call."""
    feature_rows = []
    valid_indices = []
    errors = {}

    for index, series_input in enumerate(series_inputs):
        if series_input.values.empty:
            errors[index] = "No valid numeric observations."
        else:
            try:
                features = _extract_series_features(
                    series_input.values, feature_columns
                )
                feature_rows.append(features)
                valid_indices.append(index)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                errors[index] = f"Feature extraction failed: {exc}"

        if progress_callback is not None:
            progress_callback(index + 1, len(series_inputs))

    if not feature_rows:
        return PredictionBatch(pd.DataFrame(), {}, errors)

    model_input = pd.DataFrame(feature_rows).reindex(columns=feature_columns)
    if list(model_input.columns) != feature_columns:
        raise ValueError("Could not align inference features with the model schema.")

    predictions = model.predict(model_input)
    probabilities = model.predict_proba(model_input)
    if (
        probabilities.shape != (len(valid_indices), len(model.classes_))
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("The classifier returned invalid prediction probabilities.")

    details = {}
    recommendation_rows = []
    for row_number, index in enumerate(valid_indices):
        probability_by_model = dict(
            zip(model.classes_, probabilities[row_number], strict=True)
        )
        recommendation = str(predictions[row_number])
        if recommendation not in probability_by_model:
            raise ValueError("The classifier returned an unknown model label.")

        details[index] = SeriesPrediction(
            features=feature_rows[row_number],
            recommendation=recommendation,
            confidence=float(probability_by_model[recommendation]),
            probabilities={
                str(label): float(prob)
                for label, prob in probability_by_model.items()
            },
        )
        recommendation_rows.append(
            {
                "Series ID": series_inputs[index].series_id,
                "Recommended Model": recommendation,
                "Confidence": details[index].confidence,
            }
        )

    return PredictionBatch(
        pd.DataFrame(
            recommendation_rows,
            columns=["Series ID", "Recommended Model", "Confidence"],
        ),
        details,
        errors,
    )


def summarize_predictions(recommendations: pd.DataFrame) -> PredictionSummary:
    """Aggregate model counts, confidence, and a concise batch explanation."""
    if recommendations.empty:
        raise ValueError("Cannot summarize an empty set of recommendations.")

    counts = recommendations["Recommended Model"].value_counts()
    arima_count = int(counts.get("ARIMA", 0))
    prophet_count = int(counts.get("Prophet", 0))
    total = len(recommendations)
    if arima_count == prophet_count:
        overall_model = "Tie"
        explanation = (
            f"Recommendations are evenly split: ARIMA and Prophet each lead "
            f"{arima_count} of {total} series."
        )
    else:
        overall_model = "ARIMA" if arima_count > prophet_count else "Prophet"
        leading_count = max(arima_count, prophet_count)
        explanation = (
            f"{overall_model} is the majority recommendation for "
            f"{leading_count} of {total} series "
            f"({leading_count / total:.1%})."
        )

    return PredictionSummary(
        total=total,
        arima_count=arima_count,
        prophet_count=prophet_count,
        average_confidence=float(recommendations["Confidence"].mean()),
        overall_model=overall_model,
        explanation=explanation,
    )


def compare_forecasts(series: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit both forecasters on a holdout split for one selected series."""
    dated_series = preprocess_series(series, freq="D")
    train, test = train_test_split(dated_series)

    arima_model = train_arima(train)
    arima_predictions = np.asarray(
        forecast_arima(arima_model, steps=len(test)),
        dtype=float,
    )
    prophet_model = train_prophet(train)
    prophet_predictions = np.asarray(
        forecast_prophet(prophet_model, test),
        dtype=float,
    )

    scores = {
        "ARIMA": float(rmse(test.to_numpy(), arima_predictions)),
        "Prophet": float(rmse(test.to_numpy(), prophet_predictions)),
    }
    forecasts = pd.DataFrame(
        {
            "Actual": test.to_numpy(),
            "ARIMA": arima_predictions,
            "Prophet": prophet_predictions,
        },
        index=range(1, len(test) + 1),
    )
    forecasts.index.name = "Forecast step"
    return scores, forecasts
