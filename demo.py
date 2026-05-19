from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parent

np.random.seed(42)

num_series = 5000
num_points = 1000

data = []

for i in range(num_series):

    time = np.arange(num_points)

    trend = time * np.random.uniform(0.5, 3)

    seasonality = (
        20 *
        np.sin(time / np.random.uniform(4, 10))
    )

    noise = np.random.normal(
        0,
        10,
        num_points
    )

    values = (
        500 +
        trend +
        seasonality +
        noise
    )

    row = [f"M{i+1}"] + list(values)

    data.append(row)

columns = (
    ["ID"] +
    [f"V{i+1}" for i in range(num_points)]
)

df = pd.DataFrame(
    data,
    columns=columns
)

save_path = root / "demo.csv"

df.to_csv(
    save_path,
    index=False
)

print("Synthetic dataset created")
print(f"Saved at: {save_path}")
print(f"Dataset Shape: {df.shape}")