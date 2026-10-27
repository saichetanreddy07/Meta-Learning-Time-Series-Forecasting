from pathlib import Path
import pandas as pd


def print_rmse_summary():
    """Print average ARIMA and Prophet RMSE grouped by series frequency."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "processed" / "meta_dataset.csv"
    df = pd.read_csv(path)

    summary = df.groupby("frequency")[["rmse_arima", "rmse_prophet"]].mean()
    print("Average RMSE by frequency:")
    for frequency, row in summary.iterrows():
        print(
            f"{frequency.title()}: ARIMA={row['rmse_arima']:.2f}, "
            f"Prophet={row['rmse_prophet']:.2f}"
        )


if __name__ == "__main__":
    print_rmse_summary()
