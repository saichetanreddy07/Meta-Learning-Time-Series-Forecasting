from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_rmse_comparison():
    """Plot and save average RMSE comparison by dataset frequency."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "processed" / "meta_dataset.csv"
    save_path = root / "results" / "plots" / "rmse_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    summary = df.groupby("frequency")[["rmse_arima", "rmse_prophet"]].mean()

    summary.plot(kind="bar", figsize=(8, 5))
    plt.title("ARIMA vs Prophet RMSE Comparison")
    plt.xlabel("Dataset Frequency")
    plt.ylabel("Average RMSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("[OK] RMSE comparison plot generated")


if __name__ == "__main__":
    plot_rmse_comparison()
