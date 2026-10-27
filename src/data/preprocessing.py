from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE_COLUMN_NAMES = {"date", "datetime", "timestamp", "time"}


@dataclass(frozen=True)
class TimeSeriesInput:
    """Stores one extracted time series."""

    series_id: str
    values: pd.Series
    removed_count: int


def _is_date_column(column: pd.Series, name: str) -> bool:
    """Check whether a column contains date values."""
    if name.strip().lower() in DATE_COLUMN_NAMES:
        return True

    if pd.api.types.is_datetime64_any_dtype(column):
        return True

    if not (
        pd.api.types.is_object_dtype(column)
        or pd.api.types.is_string_dtype(column)
    ):
        return False

    non_missing = column.dropna().astype(str).str.strip()
    if non_missing.empty:
        return False

    date_like = non_missing.str.contains(r"[-/:]|[A-Za-z]{3}", regex=True)
    if not date_like.any():
        return False

    parsed_dates = pd.to_datetime(non_missing, errors="coerce", format="mixed")
    return parsed_dates.notna().sum() >= 0.8 * len(non_missing)


def extract_time_series(frame: pd.DataFrame) -> list[TimeSeriesInput]:
    """
    Extract one or more time series from an uploaded CSV.

    Supported formats:
    1. Date, Value (single series)
    2. ID, V1, V2, V3... (multiple series)
    """
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")

    frame = frame.reset_index(drop=True)
    date_columns = {
        column
        for column in frame.columns
        if _is_date_column(frame[column], str(column))
    }

    observation_columns = []
    numeric_columns = {}

    for column in frame.columns:
        if column in date_columns:
            continue

        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        values = numeric_series.to_numpy(dtype=float, na_value=np.nan)

        if pd.api.types.is_numeric_dtype(frame[column]) or np.isfinite(values).any():
            observation_columns.append(column)
            numeric_columns[column] = numeric_series

    if not observation_columns:
        raise ValueError("No numeric observations were found.")

    # Single time series
    if len(observation_columns) == 1:
        column = observation_columns[0]
        numeric_series = numeric_columns[column]
        values = numeric_series.to_numpy(dtype=float, na_value=np.nan)
        valid_mask = np.isfinite(values)

        series = pd.Series(
            values[valid_mask],
            name=str(column),
            dtype=float,
        ).reset_index(drop=True)

        return [
            TimeSeriesInput(
                series_id=str(column),
                values=series,
                removed_count=len(frame) - int(valid_mask.sum()),
            )
        ]

    # Multiple time series
    identifier_columns = [
        column
        for column in frame.columns
        if column not in date_columns and column not in observation_columns
    ]

    if not identifier_columns:
        raise ValueError(
            "Multiple numeric columns were found, but no ID column exists."
        )

    id_column = identifier_columns[0]
    series_list = []

    for row in range(len(frame)):
        identifier = frame.at[row, id_column]
        if pd.isna(identifier) or not str(identifier).strip():
            series_id = f"Series {row + 1}"
        else:
            series_id = str(identifier).strip()

        row_values = np.asarray(
            [numeric_columns[col].iat[row] for col in observation_columns],
            dtype=float,
        )
        valid_mask = np.isfinite(row_values)
        series = pd.Series(row_values[valid_mask], name=series_id, dtype=float)

        series_list.append(
            TimeSeriesInput(
                series_id=series_id,
                values=series,
                removed_count=len(row_values) - int(valid_mask.sum()),
            )
        )

    return series_list


def preprocess_series(
    series: pd.Series,
    freq: str = "MS",
    start: str | pd.Timestamp = "2000-01-01",
) -> pd.Series:
    """Clean a series and assign a regular datetime index."""
    if series.empty:
        raise ValueError("Series must not be empty.")

    series = (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )

    if series.empty:
        raise ValueError("Series contains no numeric values.")

    series.index = pd.date_range(
        start=start,
        periods=len(series),
        freq=freq,
    )
    return series


def split_series(
    series: pd.Series,
    train_ratio: float = 0.8,
) -> tuple[pd.Series, pd.Series]:
    """Split an uploaded time series into train and test sets."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if len(series) < 2:
        raise ValueError("Series must contain at least two observations.")

    split_index = int(len(series) * train_ratio)
    split_index = min(max(split_index, 1), len(series) - 1)

    train = series.iloc[:split_index]
    test = series.iloc[split_index:]
    return train, test


# Alias for compatibility with inference
train_test_split = split_series
