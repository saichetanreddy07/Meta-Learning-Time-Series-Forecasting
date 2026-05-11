from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.pipeline.create_meta_dataset import create_meta_dataset
    from src.models.meta_model import train_meta_model
    from results.scripts.generate_plots import generate_plots
    from results.scripts.plot_feature_importance import plot_feature_importance

    print("Creating meta dataset...")
    create_meta_dataset()

    print("Training meta model...")
    model = train_meta_model()

    print("Generating plots...")
    generate_plots()

    print("Generating feature importance plot...")
    plot_feature_importance(model, model.feature_names_in_)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()