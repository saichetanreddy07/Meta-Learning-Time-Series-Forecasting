from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]

path = root / "data" / "processed" / "meta_dataset.csv"

df = pd.read_csv(path)

summary = df.groupby("frequency")[["rmse_arima", "rmse_prophet"]].mean()

print(summary)