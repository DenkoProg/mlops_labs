import json
from pathlib import Path
from typing import Annotated, Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import typer


matplotlib.use("Agg")

app = typer.Typer()

CLASSES = ["NORMAL", "PNEUMONIA"]


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, img_size: int, path: str, top_n: int = 20):
    importances = model.feature_importances_
    importance_map = importances.reshape(img_size, img_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(importance_map, cmap="hot")
    axes[0].set_title("Feature Importance Heatmap")
    axes[0].axis("off")

    top_indices = np.argsort(importances)[-top_n:]
    axes[1].barh(range(top_n), importances[top_indices])
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels([f"pixel_{i}" for i in top_indices])
    axes[1].set_title(f"Top {top_n} Features")
    axes[1].set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@app.command()
def main(
    data_dir: Annotated[Path, typer.Argument(help="Path to prepared data directory")],
    models_dir: Annotated[Path, typer.Argument(help="Path to output models directory")],
    n_estimators: Annotated[int, typer.Option(help="Number of trees in the forest")] = 100,
    max_depth: Annotated[int | None, typer.Option(help="Maximum depth of each tree")] = None,
    min_samples_split: Annotated[int, typer.Option(help="Minimum samples required to split a node")] = 2,
    class_weight: Annotated[str | None, typer.Option(help="Class weight strategy: 'balanced' or None")] = "balanced",
    img_size: Annotated[int, typer.Option(help="Original img_size for feature importance plotting")] = 64,
    random_state: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    experiment_name: Annotated[
        str | None, typer.Option(help="MLflow experiment name (auto-generated if not set)")
    ] = None,
):
    typer.echo(f"Loading prepared training data from {data_dir}...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    typer.echo(f"Loading prepared test data from {data_dir}...")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    typer.echo(f"Train: {X_train.shape}, Test: {X_test.shape}")
    typer.echo(f"Train distribution: NORMAL={np.sum(y_train == 0)}, PNEUMONIA={np.sum(y_train == 1)}")

    sklearn_class_weight = class_weight if class_weight in ("balanced", "balanced_subsample") else None

    # Auto-generate experiment name if not provided
    if experiment_name is None:
        depth_str = f"maxdepth{max_depth}" if max_depth is not None else "maxdepthNone"
        experiment_name = f"rf_chest_xray_{depth_str}_nest{n_estimators}"

    typer.echo(f"📊 Experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset_version", "chest_xray_v1_dvc")
        mlflow.set_tag("img_size", str(img_size))
        mlflow.set_tag("class_weight", str(class_weight))

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "class_weight": str(sklearn_class_weight),
            "img_size": img_size,
            "random_state": random_state,
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=sklearn_class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

        typer.echo("Training model...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "train_recall": recall_score(y_train, y_train_pred),
        }
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
        }

        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)

        # Save metrics manually for DVC outputs
        metrics_dict = {"train": train_metrics, "test": test_metrics}
        metrics_file = models_dir / "metrics.json"
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        typer.echo("\n--- Train Metrics ---")
        for k, v in train_metrics.items():
            typer.echo(f"  {k}: {v:.4f}")
        typer.echo("\n--- Test Metrics ---")
        for k, v in test_metrics.items():
            typer.echo(f"  {k}: {v:.4f}")

        typer.echo("\n" + classification_report(y_test, y_test_pred, target_names=CLASSES))

        artifacts_dir = Path(f"artifacts/{experiment_name}")
        artifacts_dir.mkdir(exist_ok=True, parents=True)

        cm_path = str(artifacts_dir / "confusion_matrix.png")
        save_confusion_matrix(y_test, y_test_pred, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = str(artifacts_dir / "feature_importance.png")
        save_feature_importance(model, img_size, fi_path)
        mlflow.log_artifact(fi_path)

        mlflow.sklearn.log_model(model, "random_forest_model")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        mlflow.sklearn.log_model(model, "random_forest_model")

        models_dir.mkdir(exist_ok=True, parents=True)
        model_filename = "rf_model.joblib"
        model_path = models_dir / model_filename
        joblib.dump(model, model_path)
        typer.echo(f"Model saved to {model_path}")

        typer.echo("Run logged to MLflow.")


if __name__ == "__main__":
    app()
