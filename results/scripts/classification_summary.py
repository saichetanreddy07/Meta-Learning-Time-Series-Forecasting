from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def generate_classification_summary():
    """Evaluate Random Forest classifier and save results table as an image."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "processed" / "meta_dataset.csv"
    df = pd.read_csv(path)

    X = df.drop(
        columns=["best_model", "rmse", "rmse_arima", "rmse_prophet"]
    )
    X = pd.get_dummies(X, columns=["frequency"])
    y = df["best_model"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                round(accuracy, 4),
                round(report["weighted avg"]["precision"], 4),
                round(report["weighted avg"]["recall"], 4),
                round(report["weighted avg"]["f1-score"], 4),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    save_path = root / "results" / "plots" / "classification_results_table.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[OK] Classification summary generated (accuracy: {accuracy:.2%})")


if __name__ == "__main__":
    generate_classification_summary()
