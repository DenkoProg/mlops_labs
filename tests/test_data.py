"""Pre-train tests: validate the prepared dataset before training."""

from pathlib import Path

import numpy as np
import pytest


PREPARED_DIR = Path("data/prepared")
EXPECTED_FILES = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]


@pytest.fixture(scope="module")
def prepared_data():
    data = {}
    for fname in EXPECTED_FILES:
        data[fname] = np.load(PREPARED_DIR / fname)
    return data


def test_prepared_files_exist():
    """All four .npy files must be present."""
    for fname in EXPECTED_FILES:
        assert (PREPARED_DIR / fname).exists(), f"Missing file: {PREPARED_DIR / fname}"


def test_data_shapes(prepared_data):
    """Arrays must have the correct dimensionality."""
    assert prepared_data["X_train.npy"].ndim == 2, "X_train must be 2D"
    assert prepared_data["X_test.npy"].ndim == 2, "X_test must be 2D"
    assert prepared_data["y_train.npy"].ndim == 1, "y_train must be 1D"
    assert prepared_data["y_test.npy"].ndim == 1, "y_test must be 1D"


def test_data_consistency(prepared_data):
    """Number of samples must match between X and y arrays."""
    assert prepared_data["X_train.npy"].shape[0] == prepared_data["y_train.npy"].shape[0], (
        "X_train and y_train sample count mismatch"
    )
    assert prepared_data["X_test.npy"].shape[0] == prepared_data["y_test.npy"].shape[0], (
        "X_test and y_test sample count mismatch"
    )


def test_feature_dimensions_match(prepared_data):
    """Train and test must have the same number of features."""
    assert prepared_data["X_train.npy"].shape[1] == prepared_data["X_test.npy"].shape[1], (
        "X_train and X_test have different feature dimensions"
    )


def test_label_values(prepared_data):
    """Labels must be binary: only 0 (NORMAL) and 1 (PNEUMONIA)."""
    for key in ("y_train.npy", "y_test.npy"):
        unique = set(np.unique(prepared_data[key]))
        assert unique <= {0, 1}, f"{key} contains unexpected label values: {unique}"


def test_pixel_range(prepared_data):
    """Pixel values must be normalized to [0.0, 1.0]."""
    for key in ("X_train.npy", "X_test.npy"):
        arr = prepared_data[key]
        assert arr.min() >= 0.0, f"{key} has values below 0"
        assert arr.max() <= 1.0, f"{key} has values above 1 (not normalized)"


def test_class_balance(prepared_data):
    """Each class must represent at least 10% of training data."""
    y_train = prepared_data["y_train.npy"]
    n_total = len(y_train)
    for label in (0, 1):
        ratio = np.sum(y_train == label) / n_total
        assert ratio >= 0.10, f"Class {label} is severely underrepresented: {ratio:.2%}"


def test_no_nan_values(prepared_data):
    """Data arrays must not contain NaN values."""
    for key in ("X_train.npy", "X_test.npy"):
        assert not np.isnan(prepared_data[key]).any(), f"{key} contains NaN values"


def test_minimum_dataset_size(prepared_data):
    """Dataset must have enough samples to be meaningful."""
    assert prepared_data["X_train.npy"].shape[0] >= 100, "Training set too small (< 100 samples)"
    assert prepared_data["X_test.npy"].shape[0] >= 50, "Test set too small (< 50 samples)"
