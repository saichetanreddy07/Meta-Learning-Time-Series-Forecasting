from pathlib import Path

import pandas as pd

from results.scripts.generate_plots import plot_prediction
from src.data.m4_loader import (
    load_m4_monthly_test,
    load_m4_monthly_train,
    load_m4_weekly_test,
    load_m4_weekly_train,
    load_m4_yearly_test,
    load_m4_yearly_train,
)
from src.data.preprocessing import preprocess_series
from src.evaluation.metrics import rmse
from src.features.feature_extraction import extract_meta_features
from src.models.arima_model import forecast_arima, train_arima
from src.models.prophet_model import forecast_prophet, train_prophet


def _evaluate_models(
    train: pd.Series,
    test: pd.Series,
    frequency: str,
    series_number: int,
) -> tuple[dict, str, float, float, float, tuple | None]:
    """Train ARIMA and Prophet on training series, evaluate RMSE on test series."""
    features = extract_meta_features(train)

    # ARIMA forecast and RMSE
    model_arima = train_arima(train)
    preds_arima = forecast_arima(model_arima, steps=len(test))
    rmse_arima = float(rmse(test.values, preds_arima))

    # Prophet forecast and RMSE
    plot_data = None
    try:
        model_prophet = train_prophet(train)
        preds_prophet = forecast_prophet(model_prophet, test)
        rmse_prophet = float(rmse(test.values, preds_prophet))
        plot_data = (test.values, preds_arima, preds_prophet)
    except Exception as exc:
        print(
            f"[WARN] Prophet failed for {frequency.title()} Series "
            f"({series_number}): {exc}"
        )
        rmse_prophet = float("inf")

    # Select the model with lower RMSE
    if rmse_arima < rmse_prophet:
        best_model = "ARIMA"
        best_rmse = rmse_arima
    else:
        best_model = "Prophet"
        best_rmse = rmse_prophet

    return features, best_model, best_rmse, rmse_arima, rmse_prophet, plot_data


def create_meta_dataset(max_series: int = 500) -> pd.DataFrame:
    """Compare forecasters on official M4 train/test pairs and save meta-features."""
    if max_series < 1:
        raise ValueError("max_series must be at least 1")

    frequencies = {
        "monthly": (
            "MS",
            lambda: load_m4_monthly_train(max_series=max_series),
            lambda: load_m4_monthly_test(max_series=max_series),
        ),
        "weekly": (
            "W",
            lambda: load_m4_weekly_train(max_series=max_series),
            lambda: load_m4_weekly_test(max_series=max_series),
        ),
        "yearly": (
            "YS",
            lambda: load_m4_yearly_train(max_series=max_series),
            lambda: load_m4_yearly_test(max_series=max_series),
        ),
    }

    records = []
    forecast_plot_data = None

    for frequency, (date_frequency, load_train, load_test) in frequencies.items():
        train_series = load_train()
        test_series = load_test()

        if len(train_series) != len(test_series):
            print(
                f"[WARN] {frequency.title()} train/test counts differ "
                f"({len(train_series)} vs {len(test_series)}); "
                "only matching row positions will be considered."
            )

        pair_count = min(len(train_series), len(test_series), max_series)
        for index in range(pair_count):
            series_number = index + 1
            print(
                f"Processing {frequency.title()} Series "
                f"({series_number}/{pair_count})"
            )

            raw_train = train_series[index]
            raw_test = test_series[index]
            if raw_train.empty or raw_test.empty:
                print(
                    f"[WARN] Skipping {frequency.title()} Series "
                    f"({series_number}): train or test series is empty."
                )
                continue

            try:
                train = preprocess_series(raw_train, freq=date_frequency)
                test_start = pd.date_range(
                    start=train.index[-1],
                    periods=2,
                    freq=date_frequency,
                )[-1]
                test = preprocess_series(
                    raw_test,
                    freq=date_frequency,
                    start=test_start,
                )

                if train.empty or test.empty:
                    print(
                        f"[WARN] Skipping {frequency.title()} Series "
                        f"({series_number}): train or test series is empty."
                    )
                    continue

                (
                    features,
                    best_model,
                    best_rmse,
                    rmse_arima,
                    rmse_prophet,
                    sample_plot,
                ) = _evaluate_models(train, test, frequency, series_number)

                if forecast_plot_data is None and sample_plot is not None:
                    forecast_plot_data = sample_plot

                records.append({
                    "frequency": frequency,
                    **features,
                    "best_model": best_model,
                    "rmse": best_rmse,
                    "rmse_arima": rmse_arima,
                    "rmse_prophet": rmse_prophet,
                })
                print(f"[OK] Best Model: {best_model} | RMSE: {best_rmse:.2f}")

            except Exception as exc:
                print(
                    f"[WARN] Skipping {frequency.title()} Series "
                    f"({series_number}): {exc}"
                )

    if not records:
        raise ValueError("No valid M4 train/test pairs produced meta-data")

    df = pd.DataFrame(records)
    root = Path(__file__).resolve().parents[2]
    save_path = root / "data" / "processed" / "meta_dataset.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    if forecast_plot_data is not None:
        plot_prediction(*forecast_plot_data)

    return df
