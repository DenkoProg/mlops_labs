"""
Airflow DAG: ML Training Pipeline (Continuous Training)

Orchestrates the full ML lifecycle:
  prepare_data → train_model → evaluate_model → register_model

Scheduled @weekly to implement Continuous Training (CT).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator


# Project root is mounted at /opt/mlops inside the Airflow container,
# but when testing locally we fall back to the repo root.
PROJECT_ROOT = os.environ.get("MLOPS_PROJECT_ROOT", ".")
UV_RUN = f"cd {PROJECT_ROOT} && .venv/bin/python"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ml_training_pipeline",
    description="Chest X-Ray Pneumonia — Continuous Training pipeline",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "training", "chest-xray"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(f"{UV_RUN} src/prepare.py " "data/raw/chest_xray data/prepared --img-size 64"),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && .venv/bin/python src/train.py "
            "data/prepared models --n-estimators 100 --max-depth 15"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(f"cd {PROJECT_ROOT} && .venv/bin/pytest tests/test_artifacts.py -v --tb=short"),
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"{UV_RUN} src/register_model.py",
    )

    # Define DAG topology
    prepare_data >> train_model >> evaluate_model >> register_model
