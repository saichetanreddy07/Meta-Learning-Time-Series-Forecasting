def train_arima(train):
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np

    try:
        values = train.dropna().astype(float)
    except AttributeError:
        values = np.asarray(train, dtype=float)
        values = values[~np.isnan(values)]

    if len(values) < 2:
        raise ValueError("Not enough data")

    model = ARIMA(values, order=(1, 1, 1))
    return model.fit()


def forecast_arima(model, steps: int):
    if steps <= 0:
        raise ValueError("steps must be positive")

    return model.forecast(steps=steps)