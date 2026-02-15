# MLOps Lab 1 — Chest X-Ray Pneumonia Classification

Binary image classification (NORMAL vs PNEUMONIA) using chest X-ray images with full MLflow experiment tracking.

## Project Structure

```
mlops_lab_1/
├── data/raw/chest_xray/     # Dataset (not in Git)
│   ├── train/
│   ├── test/
│   └── val/
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory data analysis
├── src/
│   ├── download_data.py     # Dataset downloader
│   └── train.py             # Training script with MLflow
├── models/                  # Saved models (not in Git)
├── mlruns/                  # MLflow tracking (not in Git)
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
uv sync
```

## Usage

### 1. Download Data

Requires [Kaggle credentials](https://www.kaggle.com/docs/api#authentication) configured.

```bash
make download-data
```

### 2. Run EDA

```bash
make eda
```

### 3. Train Model

```bash
# Default parameters
make train

# Custom hyperparameters
make train ARGS="--n-estimators 200 --max-depth 15 --img-size 64"
```

#### CLI Arguments

| Argument             | Default                  | Description                 |
| -------------------- | ------------------------ | --------------------------- |
| `--n-estimators`     | 100                      | Number of trees             |
| `--max-depth`        | None                     | Max tree depth              |
| `--min-samples-split`| 2                        | Min samples to split a node |
| `--img-size`         | 64                       | Image resize dimension      |
| `--random-state`     | 42                       | Random seed                 |
| `--experiment-name`  | chest_xray_pneumonia     | MLflow experiment name      |

### 4. View Experiments

```bash
make mlflow-ui
# Open http://127.0.0.1:5000
```

### 5. Run Multiple Experiments

```bash
# 1. Underfitting: low depth
make train ARGS="--max-depth 3 --n-estimators 100"
# 2. Moderate depth
make train ARGS="--max-depth 10 --n-estimators 100"
# 3. High depth
make train ARGS="--max-depth 20 --n-estimators 100"
# 4. Very high depth
make train ARGS="--max-depth 50 --n-estimators 100"
# 5. Overfitting: unlimited depth
make train ARGS="--n-estimators 100"
```
