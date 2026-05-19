import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data.preprocess import preprocess_series, train_test_split
from src.models.base_models import train_arima, forecast_arima
from src.models.prophet_model import train_prophet, forecast_prophet
from src.evaluation.metrics import rmse

st.set_page_config(
    page_title="Meta-Learning Forecasting",
    layout="wide"
)

st.title("Meta-Learning Time Series Forecasting System")

st.write(
    """
    Upload a CSV file to compare ARIMA and Prophet forecasting models
    and automatically identify the best forecasting model.
    """
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.write(df.head())

        st.write("Dataset Shape:", df.shape)

        series = df.iloc[0, 1:]

        series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        processed = preprocess_series(series)

        train, test = train_test_split(processed)

        st.subheader("Original Time Series")

        fig1, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(
            range(len(processed)),
            processed.values
        )

        ax1.set_xlabel("Time Step")

        ax1.set_ylabel("Values")

        st.pyplot(fig1)

        model_arima = train_arima(train)

        preds_arima = forecast_arima(
            model_arima,
            steps=len(test)
        )

        preds_arima = pd.Series(
            preds_arima
        ).reset_index(drop=True)

        rmse_arima = rmse(
            test.values,
            preds_arima
        )

        model_prophet = train_prophet(train)

        preds_prophet = forecast_prophet(
            model_prophet,
            test
        )

        preds_prophet = pd.Series(
            preds_prophet
        ).reset_index(drop=True)

        rmse_prophet = rmse(
            test.values,
            preds_prophet
        )

        if rmse_arima < rmse_prophet:

            best_model = "ARIMA"

        else:

            best_model = "Prophet"

        st.subheader("Model Comparison")

        col1, col2 = st.columns(2)

        col1.metric(
            "ARIMA RMSE",
            f"{rmse_arima:.2f}"
        )

        col2.metric(
            "Prophet RMSE",
            f"{rmse_prophet:.2f}"
        )

        st.success(
            f"Best Model Selected: {best_model}"
        )

        st.subheader("Forecast Comparison")

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        x = range(len(test))

        ax2.plot(
            x,
            test.values,
            label="Actual",
            linewidth=2
        )

        ax2.plot(
            x,
            preds_arima,
            label="ARIMA Predictions",
            linestyle="--"
        )

        ax2.plot(
            x,
            preds_prophet,
            label="Prophet Predictions",
            linestyle=":"
        )

        ax2.set_xlabel("Time Step")

        ax2.set_ylabel("Values")

        ax2.legend()

        st.pyplot(fig2)

    except Exception as e:

        st.error(f"Error occurred: {e}")