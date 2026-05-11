import pandas as pd
from pathlib import Path


def train_meta_model():

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import (
        accuracy_score,
        classification_report
    )

    root = Path(__file__).resolve().parents[2]

    path = (
        root /
        "data" /
        "processed" /
        "meta_dataset.csv"
    )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("meta_dataset.csv is empty")

    if "best_model" not in df.columns:
        raise KeyError("'best_model' column missing")

    y = df["best_model"]

    X = df.drop(
        columns=[
            "best_model",
            "rmse",
            "rmse_arima",
            "rmse_prophet"
        ],
        errors="ignore"
    )

    X = pd.get_dummies(
        X,
        columns=["frequency"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\nMeta Model Accuracy: {acc:.4f}\n")

    print("Classification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    importance = pd.DataFrame({

        "feature": X.columns,

        "importance": model.feature_importances_

    }).sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop Features:")

    print(importance.head())

    return model