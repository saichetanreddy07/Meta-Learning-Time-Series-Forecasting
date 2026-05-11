from pathlib import Path
import pandas as pd


def load_dataset(file_name):

    root = Path(__file__).resolve().parents[2]

    path = root / "data" / "raw" / "train" / file_name

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)

    series_list = []

    for _, row in df.iloc[:, 1:].iterrows():

        values = pd.to_numeric(
            row,
            errors="coerce"
        ).dropna()

        series_list.append(values)

    return series_list


def load_m4_monthly():
    return load_dataset("Monthly-train.csv")


def load_m4_weekly():
    return load_dataset("Weekly-train.csv")


def load_m4_yearly():
    return load_dataset("Yearly-train.csv")