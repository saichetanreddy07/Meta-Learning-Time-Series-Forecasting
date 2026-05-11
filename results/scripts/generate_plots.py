from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction(actual, pred_arima, pred_prophet):

    root = Path(__file__).resolve().parents[2]

    save_dir = (
        root /
        "results" /
        "plots"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = (
        save_dir /
        "arima_vs_prophet.png"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        actual,
        label="Actual"
    )

    plt.plot(
        pred_arima,
        label="ARIMA"
    )

    plt.plot(
        pred_prophet,
        label="Prophet"
    )

    plt.xlabel("Time Step")

    plt.ylabel("Value")

    plt.title("ARIMA vs Prophet Forecast Comparison")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

    print(f"Plot saved to: {save_path}")


def generate_plots():

    root = Path(__file__).resolve().parents[2]

    csv_path = (
        root /
        "data" /
        "processed" /
        "meta_dataset.csv"
    )

    save_dir = (
        root /
        "results" /
        "plots"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(csv_path)

    # Best model distribution
    plt.figure(figsize=(6, 4))

    df["best_model"].value_counts().plot(
        kind="bar"
    )

    plt.title("Best Model Distribution")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        save_dir /
        "best_model_distribution.png"
    )

    plt.close()

    # RMSE comparison
    plt.figure(figsize=(10, 5))

    plt.plot(
        df["rmse_arima"].values,
        label="ARIMA"
    )

    plt.plot(
        df["rmse_prophet"].values,
        label="Prophet"
    )

    plt.title("RMSE Comparison")

    plt.xlabel("Series")

    plt.ylabel("RMSE")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_dir /
        "rmse_comparison.png"
    )

    plt.close()

    print("Plots generated successfully")


if __name__ == "__main__":

    import numpy as np

    actual = np.random.rand(20)

    arima = np.random.rand(20)

    prophet = np.random.rand(20)

    plot_prediction(
        actual,
        arima,
        prophet
    )

    generate_plots()