import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Print model evaluation metrics."""

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    labels = list(model.classes_)

    print("\n" + "=" * 60)
    print("Random Forest Evaluation")
    print("=" * 60)

    print(f"Train Accuracy : {accuracy_score(y_train, train_predictions):.2%}")
    print(f"Test Accuracy  : {accuracy_score(y_test, test_predictions):.2%}")

    print("\nClass Distribution")
    print(y_train.value_counts().to_dict())

    print("\nConfusion Matrix")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, test_predictions, labels=labels),
            index=[f"True_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels],
        )
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    probability_df = pd.DataFrame(
        probabilities,
        columns=[f"{label}_probability" for label in labels],
    )

    print("\nPrediction Probabilities (First 5)")
    print(probability_df.head())


def save_model(model, feature_columns, root):
    """Save trained model and feature names."""

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "random_forest.pkl")

    with open(model_dir / "feature_columns.json", "w") as file:
        json.dump(feature_columns, file, indent=2)

    print("\n[OK] Model Saved")


def save_feature_importance(model, feature_columns, root):
    """Save feature importance."""

    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
    )

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    importance.to_csv(
        results_dir / "feature_importance.csv",
        index=False,
    )

    print("[OK] Feature Importance Saved")


def train_meta_model() -> RandomForestClassifier:
    """Train and save the Random Forest meta-model."""

    root = Path(__file__).resolve().parents[2]

    meta_dataset_path = (
        root
        / "data"
        / "processed"
        / "meta_dataset.csv"
    )

    df = pd.read_csv(meta_dataset_path)

    if df.empty:
        raise ValueError("meta_dataset.csv is empty")

    if "best_model" not in df.columns:
        raise KeyError("'best_model' column missing")

    X = df.drop(
        columns=[
            "frequency",
            "best_model",
            "rmse",
            "rmse_arima",
            "rmse_prophet",
        ],
        errors="ignore",
    )

    y = df["best_model"]

    print("\nDataset Summary")
    print(f"Samples : {len(df)}")
    print(f"Classes : {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
    )

    model.fit(X_train, y_train)

    evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Retrain on the complete dataset before saving
    model.fit(X, y)

    save_model(
        model,
        list(X.columns),
        root,
    )

    save_feature_importance(
        model,
        list(X.columns),
        root,
    )

    return model