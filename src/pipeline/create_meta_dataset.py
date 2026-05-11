import pandas as pd
from pathlib import Path


def create_meta_dataset():

    from src.data.load_m4 import (
        load_m4_monthly,
        load_m4_weekly,
        load_m4_yearly
    )

    from src.data.preprocess import (
        preprocess_series,
        train_test_split
    )

    from src.features.meta_features import (
        extract_meta_features
    )

    from src.evaluation.metrics import rmse

    from src.models.base_models import (
        train_arima,
        forecast_arima
    )

    from src.models.prophet_model import (
        train_prophet,
        forecast_prophet
    )

    records = []

    monthly = load_m4_monthly()
    weekly = load_m4_weekly()
    yearly = load_m4_yearly()

    series_list = (
        [(s, "monthly") for s in monthly[:20]] +
        [(s, "weekly") for s in weekly[:20]] +
        [(s, "yearly") for s in yearly[:20]]
    )

    for i, (series, freq) in enumerate(series_list):

        print(f"Processing series {i} ({freq})")

        try:

            processed = preprocess_series(series)

            train, test = train_test_split(
                processed,
                ratio=0.8
            )

            if len(test) == 0:
                continue

            features = extract_meta_features(processed)

            model_arima = train_arima(train)

            preds_arima = forecast_arima(
                model_arima,
                steps=len(test)
            )

            rmse_arima = float(
                rmse(test.values, preds_arima)
            )

            try:

                model_prophet = train_prophet(train)

                preds_prophet = forecast_prophet(
                    model_prophet,
                    test
                )

                rmse_prophet = float(
                    rmse(test.values, preds_prophet)
                )

            except Exception:

                rmse_prophet = float("inf")

            if rmse_arima < rmse_prophet:

                best_model = "ARIMA"
                best_rmse = rmse_arima

            else:

                best_model = "Prophet"
                best_rmse = rmse_prophet

            records.append({

                "frequency": freq,

                **features,

                "best_model": best_model,

                "rmse": best_rmse,

                "rmse_arima": rmse_arima,

                "rmse_prophet": rmse_prophet
            })

            print(
                f"ARIMA={rmse_arima:.2f}, "
                f"Prophet={rmse_prophet:.2f} "
                f"-> {best_model}"
            )

        except Exception as e:

            print(f"Skipped series {i}: {e}")

    print(f"Total records: {len(records)}")

    if len(records) == 0:
        raise ValueError("No data created")

    df = pd.DataFrame(records)

    root = Path(__file__).resolve().parents[2]

    save_path = (
        root /
        "data" /
        "processed" /
        "meta_dataset.csv"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        save_path,
        index=False
    )

    return df