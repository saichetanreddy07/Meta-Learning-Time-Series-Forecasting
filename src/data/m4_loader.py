from pathlib import Path
import pandas as pd

# Default number of time series to load from each frequency
TRAIN_SERIES_PER_FREQUENCY = 500


def load_dataset(
    file_name: str,
    split: str = "train",
    max_series: int | None = None,
) -> list[pd.Series]:
    """Load time series from an M4 train or test CSV file."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    if max_series is not None and max_series < 1:
        raise ValueError("max_series must be at least 1")

    root = Path(__file__).resolve().parents[2]
    file_path = root / "data" / "raw" / split / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    try:
        df = pd.read_csv(file_path, nrows=max_series)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty: {file_path}")

    if df.empty or df.shape[1] < 2:
        raise ValueError(f"No time series found in {file_path}")

    series_list = []
    # Ignore the first column (series ID)
    for _, row in df.iloc[:, 1:].iterrows():
        series = (
            pd.to_numeric(row, errors="coerce")
            .dropna()
            .reset_index(drop=True)
        )
        if not series.empty:
            series_list.append(series)

    if not series_list:
        raise ValueError(f"No valid time series found in {file_path}")

    return series_list


# -------------------------
# Monthly
# -------------------------

def load_m4_monthly_train(
    max_series: int | None = TRAIN_SERIES_PER_FREQUENCY,
) -> list[pd.Series]:
    return load_dataset("Monthly-train.csv", "train", max_series)


def load_m4_monthly_test(
    max_series: int | None = TRAIN_SERIES_PER_FREQUENCY,
) -> list[pd.Series]:
    return load_dataset("Monthly-test.csv", "test", max_series)


# -------------------------
# Weekly
# -------------------------

def load_m4_weekly_train(
    max_series: int | None = TRAIN_SERIES_PER_FREQUENCY,
) -> list[pd.Series]:
    return load_dataset("Weekly-train.csv", "train", max_series)


def load_m4_weekly_test(
    max_series: int | None = TRAIN_SERIES_PER_FREQUENCY,
) -> list[pd.Series]:
    return load_dataset("Weekly-test.csv", "test", max_series)


# -------------------------
# Yearly
# -------------------------

def load_m4_yearly_train(
    max_series: int | None = TRAIN_SERIES_PER_FREQUENCY,
) -> list[pd.Series]:
    return load_dataset("Yearly-train.csv", "train", max_series)


def load_m4_yearly_test(
    max_series: int | None = TRAIN_SERIES_PER_FREQUENCY,
) -> list[pd.Series]:
    return load_dataset("Yearly-test.csv", "test", max_series)
