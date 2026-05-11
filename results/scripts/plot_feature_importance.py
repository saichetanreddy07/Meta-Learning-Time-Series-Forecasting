from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):

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
        "feature_importance.png"
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": importances

    }).sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel("Importance")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

    print(f"Feature importance plot saved to: {save_path}")


if __name__ == "__main__":

    from src.models.meta_model import train_meta_model

    model = train_meta_model()

    feature_names = model.feature_names_in_

    plot_feature_importance(
        model,
        feature_names
    )

    print("Feature importance plot generated")