from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):
    """Plot and save feature importances of the Random Forest model."""
    root = Path(__file__).resolve().parents[2]
    save_dir = root / "results" / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "feature_importance.png"

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    from src.models.random_forest_meta_model import train_meta_model

    model = train_meta_model()
    plot_feature_importance(model, model.feature_names_in_)
