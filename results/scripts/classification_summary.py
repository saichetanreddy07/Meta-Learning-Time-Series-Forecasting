from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Project root
root = Path(__file__).resolve().parents[2]

# Load meta dataset
path = root / "data" / "processed" / "meta_dataset.csv"

df = pd.read_csv(path)

# Features and labels
X = df.drop(columns=[
    "best_model",
    "rmse",
    "rmse_arima",
    "rmse_prophet"
])

# Convert categorical frequency column
X = pd.get_dummies(X, columns=["frequency"])

y = df["best_model"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

# Summary dataframe
summary = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],
    "Value": [
        round(accuracy, 4),
        round(report["weighted avg"]["precision"], 4),
        round(report["weighted avg"]["recall"], 4),
        round(report["weighted avg"]["f1-score"], 4),
    ]
})

# Print table
print(summary)

# Generate table image
fig, ax = plt.subplots(figsize=(5, 2))

ax.axis('off')

table = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Save image
save_path = root / "results" / "plots" / "classification_results_table.png"

save_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    save_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()

print(f"Classification table saved at: {save_path}")