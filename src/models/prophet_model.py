import logging

import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress routine CmdStan logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def train_prophet(train_series: pd.Series) -> Prophet:
    """Train a Prophet model on a time series."""
    if train_series.empty:
        raise ValueError("The training series must not be empty.")

    train_df = pd.DataFrame(
        {
            "ds": train_series.index,
            "y": train_series.values,
        }
    )
    model = Prophet()
    model.fit(train_df)
    return model


def forecast_prophet(
    model: Prophet,
    test_series: pd.Series,
) -> np.ndarray:
    """Forecast future values using a trained Prophet model."""
    if test_series.empty:
        raise ValueError("The test series must not be empty.")

    future = pd.DataFrame(
        {
            "ds": test_series.index,
        }
    )
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy()
