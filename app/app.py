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


st.set_page_config(page_title="Meta-Learning Forecasting", layout="wide")

st.title("Meta-Learning Time Series Forecasting System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    series = df.iloc[:, 0]

    processed = preprocess_series(series)

    train, test = train_test_split(processed)

    st.subheader("Original Time Series")

    fig1, ax1 = plt.subplots()
    ax1.plot(processed.values)
    st.pyplot(fig1)

    # ARIMA
    model_arima = train_arima(train)
    preds_arima = forecast_arima(model_arima, steps=len(test))
    rmse_arima = rmse(test.values, preds_arima)

    # Prophet
    model_prophet = train_prophet(train)
    preds_prophet = forecast_prophet(model_prophet, test)
    rmse_prophet = rmse(test.values, preds_prophet)

    # Best model
    if rmse_arima < rmse_prophet:
        best_model = "ARIMA"
    else:
        best_model = "Prophet"

    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    col1.metric("ARIMA RMSE", f"{rmse_arima:.2f}")
    col2.metric("Prophet RMSE", f"{rmse_prophet:.2f}")

    st.success(f"Best Model Selected: {best_model}")

    st.subheader("Forecast Comparison")

    fig2, ax2 = plt.subplots()

    ax2.plot(test.values, label="Actual")
    ax2.plot(preds_arima, label="ARIMA")
    ax2.plot(preds_prophet, label="Prophet")

    ax2.legend()

    st.pyplot(fig2)