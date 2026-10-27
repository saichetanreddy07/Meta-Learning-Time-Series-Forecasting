from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_best_model_distribution():
    """Plot and save the distribution of best forecasting models."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "processed" / "meta_dataset.csv"
    save_path = root / "results" / "plots" / "best_model_distribution.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    counts = df["best_model"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Best Model Distribution")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Best model distribution plot saved")


if __name__ == "__main__":
    plot_best_model_distribution()
