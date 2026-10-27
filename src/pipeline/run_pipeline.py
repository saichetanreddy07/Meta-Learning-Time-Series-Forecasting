from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from results.scripts.generate_plots import generate_plots
from results.scripts.plot_feature_importance import plot_feature_importance
from src.models.random_forest_meta_model import train_meta_model
from src.pipeline.build_meta_dataset import create_meta_dataset


def main():
    print("=" * 50)
    print("Meta-Learning Time Series Forecasting Pipeline")
    print("=" * 50)

    print("\n[1/4] Creating Meta Dataset...\n")
    meta_dataset = create_meta_dataset()
    print(f"\n[OK] Meta Dataset Created ({len(meta_dataset)} records)")

    print("\n[2/4] Training Meta Model...")
    model = train_meta_model()

    print("\n[3/4] Generating Visualizations...")
    generate_plots()
    plot_feature_importance(model, model.feature_names_in_)
    print("[OK] Forecast Comparison")
    print("[OK] RMSE Comparison")
    print("[OK] Best Model Distribution")
    print("[OK] Feature Importance")

    print("\n[4/4] Pipeline Completed Successfully.")
    print("\n" + "=" * 50)
    print("Output Files")
    print("=" * 50)
    print("[OK] data/processed/meta_dataset.csv")
    print("[OK] models/random_forest.pkl")
    print("[OK] models/feature_columns.json")
    print("[OK] results/feature_importance.csv")
    print("[OK] results/plots/")


if __name__ == "__main__":
    main()
