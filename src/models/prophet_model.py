from prophet import Prophet
import pandas as pd


def train_prophet(train_series):
    df = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values
    })

    model = Prophet()
    model.fit(df)

    return model


def forecast_prophet(model, test_series):
    future = pd.DataFrame({
        "ds": test_series.index
    })

    forecast = model.predict(future)

    return forecast["yhat"].values