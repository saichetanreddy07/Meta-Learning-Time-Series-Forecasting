from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root
root = Path(__file__).resolve().parents[2]

# Load meta dataset
path = root / "data" / "processed" / "meta_dataset.csv"

df = pd.read_csv(path)

# Count best models
counts = df["best_model"].value_counts()

# Plot
plt.figure(figsize=(6, 6))

plt.pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Best Model Distribution")

# Save graph
save_path = (
    root /
    "results" /
    "plots" /
    "best_model_distribution.png"
)

save_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Best model distribution graph saved at: {save_path}"
)