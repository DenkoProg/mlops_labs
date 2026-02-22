"""Post-train tests: validate artifacts and enforce the Quality Gate."""

import json
from pathlib import Path

import pytest


MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = MODELS_DIR / "metrics.json"

# ── Quality Gate thresholds ──────────────────────────────────────────────────
F1_THRESHOLD = 0.80
ACCURACY_THRESHOLD = 0.75


# ── Artifact existence ───────────────────────────────────────────────────────


def test_model_file_exists():
    """At least one .joblib model file must be present in models/."""
    joblib_files = list(MODELS_DIR.glob("*.joblib"))
    assert len(joblib_files) > 0, f"No .joblib model file found in {MODELS_DIR}"


def test_metrics_json_exists():
    """models/metrics.json must exist."""
    assert METRICS_PATH.exists(), f"metrics.json not found at {METRICS_PATH}"


def test_metrics_json_valid():
    """metrics.json must be valid JSON with required keys."""
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert "test" in metrics, "metrics.json missing 'test' key"
    assert "test_f1" in metrics["test"], "metrics.json missing 'test.test_f1'"
    assert "test_accuracy" in metrics["test"], "metrics.json missing 'test.test_accuracy'"


def test_confusion_matrix_exists():
    """At least one confusion_matrix.png must exist inside artifacts/."""
    cm_files = list(ARTIFACTS_DIR.rglob("confusion_matrix.png"))
    assert len(cm_files) > 0, f"No confusion_matrix.png found under {ARTIFACTS_DIR}"


def test_feature_importance_exists():
    """At least one feature_importance.png must exist inside artifacts/."""
    fi_files = list(ARTIFACTS_DIR.rglob("feature_importance.png"))
    assert len(fi_files) > 0, f"No feature_importance.png found under {ARTIFACTS_DIR}"


# ── Quality Gate ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_quality_gate_f1(metrics):
    """Quality Gate: test F1-score must be >= threshold."""
    f1 = metrics["test"]["test_f1"]
    assert f1 >= F1_THRESHOLD, (
        f"Quality Gate FAILED: test_f1={f1:.4f} < threshold={F1_THRESHOLD}. " "Model is not good enough to proceed."
    )


def test_quality_gate_accuracy(metrics):
    """Quality Gate: test accuracy must be >= threshold."""
    acc = metrics["test"]["test_accuracy"]
    assert acc >= ACCURACY_THRESHOLD, f"Quality Gate FAILED: test_accuracy={acc:.4f} < threshold={ACCURACY_THRESHOLD}."


def test_no_severe_overfitting(metrics):
    """Train-test F1 gap must not exceed 0.15 (overfitting check)."""
    train_f1 = metrics["train"]["train_f1"]
    test_f1 = metrics["test"]["test_f1"]
    gap = train_f1 - test_f1
    assert gap <= 0.15, f"Severe overfitting detected: train_f1={train_f1:.4f}, test_f1={test_f1:.4f}, gap={gap:.4f}"
