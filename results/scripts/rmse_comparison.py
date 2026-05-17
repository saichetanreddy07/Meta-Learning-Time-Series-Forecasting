from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root
root = Path(__file__).resolve().parents[2]

# Load meta dataset
path = root / "data" / "processed" / "meta_dataset.csv"

df = pd.read_csv(path)

# Average RMSE by frequency
summary = df.groupby("frequency")[["rmse_arima", "rmse_prophet"]].mean()

# Plot
summary.plot(kind="bar", figsize=(8, 5))

plt.title("ARIMA vs Prophet RMSE Comparison")
plt.xlabel("Dataset Frequency")
plt.ylabel("Average RMSE")
plt.xticks(rotation=0)

# Save plot
save_path = root / "results" / "plots" / "rmse_comparison.png"

save_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(save_path)

plt.show()

print("RMSE comparison graph generated.")