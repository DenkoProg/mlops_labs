"""
Structural tests for the Airflow DAG.

These tests run WITHOUT a live Airflow instance — they just import
the DAG module and inspect its structure. Fast and CI-friendly.
"""

import importlib.util
from pathlib import Path
import sys

import pytest


# Load the DAG module without executing it in an Airflow context
DAG_PATH = Path("dags/ml_pipeline_dag.py")
DAG_ID = "ml_training_pipeline"
EXPECTED_TASK_IDS = {"prepare_data", "train_model", "evaluate_model", "register_model"}


@pytest.fixture(scope="module")
def dag():
    """Import the DAG module and return the DAG object."""
    spec = importlib.util.spec_from_file_location("ml_pipeline_dag", DAG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dag


def test_dag_file_exists():
    """The DAG file must be present."""
    assert DAG_PATH.exists(), f"DAG file not found: {DAG_PATH}"


def test_dag_import(dag):
    """DAG module must import without errors."""
    assert dag is not None


def test_dag_id(dag):
    """DAG must have the expected ID."""
    assert dag.dag_id == DAG_ID


def test_task_count(dag):
    """DAG must contain exactly 4 tasks."""
    assert len(dag.tasks) == 4, f"Expected 4 tasks, got {len(dag.tasks)}: {[t.task_id for t in dag.tasks]}"


def test_task_ids(dag):
    """All expected task IDs must be present."""
    actual = {t.task_id for t in dag.tasks}
    assert actual == EXPECTED_TASK_IDS, f"Task ID mismatch: {actual}"


def test_task_dependencies(dag):
    """Tasks must be wired in the correct sequential order."""
    task_map = {t.task_id: t for t in dag.tasks}

    assert task_map["train_model"].upstream_task_ids == {"prepare_data"}
    assert task_map["evaluate_model"].upstream_task_ids == {"train_model"}
    assert task_map["register_model"].upstream_task_ids == {"evaluate_model"}


def test_no_cycles(dag):
    """DAG must be acyclic (basic property of a valid DAG)."""
    # Airflow DAG raises if cycles exist at init time;
    # this test gives an explicit assertion message.
    visited = set()

    def dfs(task_id, path):
        assert task_id not in path, f"Cycle detected involving task: {task_id}"
        path = path | {task_id}
        task_map = {t.task_id: t for t in dag.tasks}
        for downstream in task_map[task_id].downstream_task_ids:
            dfs(downstream, path)
        visited.add(task_id)

    for task in dag.tasks:
        if not task.upstream_task_ids:  # start from root tasks
            dfs(task.task_id, set())


def test_dag_schedule(dag):
    """DAG should be scheduled @weekly for Continuous Training."""
    # Airflow 2.4+ uses `schedule` instead of `schedule_interval`
    schedule = getattr(dag, "schedule", None) or getattr(dag, "schedule_interval", None)
    assert schedule == "@weekly", f"Expected @weekly schedule, got: {schedule}"
