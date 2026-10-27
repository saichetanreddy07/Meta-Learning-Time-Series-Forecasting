import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ARIMA_ORDER = (1, 1, 1)


def train_arima(train: pd.Series):
    """Train an ARIMA model on a time series."""
    train = (
        pd.to_numeric(train, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )

    if len(train) < 2:
        raise ValueError("The series must contain at least two numeric values.")

    model = ARIMA(train, order=ARIMA_ORDER)
    return model.fit()


def forecast_arima(model, steps: int) -> pd.Series:
    """Generate forecasts using a trained ARIMA model."""
    if steps <= 0:
        raise ValueError("steps must be greater than 0.")

    return model.forecast(steps=steps)
