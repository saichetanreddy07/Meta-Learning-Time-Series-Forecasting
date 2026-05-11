import pandas as pd


def preprocess_series(series: pd.Series) -> pd.Series:
    series = series.reset_index(drop=True)

    series.index = pd.date_range(
        start="2000-01-01",
        periods=len(series),
        freq="MS"
    )

    return series


def train_test_split(series: pd.Series, ratio: float = 0.8):
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")

    split = int(len(series) * ratio)

    train = series.iloc[:split]
    test = series.iloc[split:]

    return train, test