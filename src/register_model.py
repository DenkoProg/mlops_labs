"""
Model registration script.

Reads models/metrics.json, compares test_f1 with the currently
registered MLflow model version, and promotes the new model if it
is better (or if no version exists yet).
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
from mlflow.exceptions import MlflowException
import mlflow.sklearn


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

METRICS_PATH = Path("models/metrics.json")
MODEL_NAME = "chest_xray_rf"
REGISTERED_MODEL_ALIAS = "champion"


def get_current_champion_f1(client: mlflow.tracking.MlflowClient) -> float:
    """Return the test_f1 of the current champion model, or 0 if none exists."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, REGISTERED_MODEL_ALIAS)
        run = client.get_run(version.run_id)
        return run.data.metrics.get("test_f1", 0.0)
    except MlflowException:
        return 0.0


def main() -> None:
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    new_f1 = metrics["test"]["test_f1"]
    new_accuracy = metrics["test"]["test_accuracy"]
    log.info(f"New model: test_f1={new_f1:.4f}, test_accuracy={new_accuracy:.4f}")

    client = mlflow.tracking.MlflowClient()

    current_f1 = get_current_champion_f1(client)
    log.info(f"Current champion: test_f1={current_f1:.4f}")

    if new_f1 <= current_f1:
        log.info(
            f"New model (f1={new_f1:.4f}) is not better than champion (f1={current_f1:.4f}). " "Skipping registration."
        )
        return

    # Load and log the model to MLflow
    model_files = list(Path("models").glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError("No .joblib model file found in models/")

    model_path = sorted(model_files)[-1]
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="model_registration") as run:
        mlflow.log_metrics({"test_f1": new_f1, "test_accuracy": new_accuracy})
        mlflow.sklearn.log_model(model, name="rf_model", registered_model_name=MODEL_NAME)
        run_id = run.info.run_id

    # Get the latest registered version and promote it
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]

    client.set_registered_model_alias(MODEL_NAME, REGISTERED_MODEL_ALIAS, latest.version)
    log.info(
        f"✅ Registered {MODEL_NAME} v{latest.version} as '{REGISTERED_MODEL_ALIAS}' "
        f"(f1: {current_f1:.4f} → {new_f1:.4f})"
    )


if __name__ == "__main__":
    main()
