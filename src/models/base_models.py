def train_arima(train):

    from statsmodels.tsa.arima.model import ARIMA
    import pandas as pd
    import numpy as np

    try:
        # Convert safely to numeric
        values = pd.to_numeric(
            train,
            errors="coerce"
        ).dropna()

    except AttributeError:

        values = np.asarray(
            train,
            dtype=float
        )

        values = values[
            ~np.isnan(values)
        ]

    # Ensure enough values
    if len(values) < 2:
        raise ValueError(
            "Not enough numeric data"
        )

    # Train ARIMA
    model = ARIMA(
        values,
        order=(1, 1, 1)
    )

    return model.fit()


def forecast_arima(model, steps: int):

    if steps <= 0:
        raise ValueError(
            "steps must be positive"
        )

    return model.forecast(
        steps=steps
    )