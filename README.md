# MLOps Labs — Chest X-Ray Pneumonia Classification

> **Binary image classification** (NORMAL vs PNEUMONIA) using chest X-ray images.
> Covers the full MLOps lifecycle across 5 labs: experiment tracking → data versioning → HPO → CI/CD → orchestration.

---

## Project Overview

| Lab | Focus | Key Tools |
|-----|-------|-----------|
| ЛР1 | Baseline training + experiment tracking | scikit-learn, MLflow |
| ЛР2 | Reproducible pipelines + data versioning | DVC, DAGsHub remote |
| ЛР3 | Config management + hyperparameter optimisation | Hydra, Optuna |
| ЛР4 | CI/CD + automated testing + CML reports | GitHub Actions, CML, pytest |
| ЛР5 | Workflow orchestration + containerisation | Apache Airflow, Docker |

---

## Project Structure

```
mlops_labs/
├── .github/
│   └── workflows/
│       └── cml.yaml              # CI/CD pipeline (lint → test → train → CML report)
├── conf/
│   └── config.yaml               # Hydra config (model params, paths)
├── dags/
│   └── ml_pipeline_dag.py        # Airflow DAG: prepare → train → evaluate → register
├── data/
│   ├── raw/chest_xray/           # Raw dataset (DVC-tracked, not in Git)
│   └── prepared/                 # Preprocessed .npy arrays (DVC-tracked)
├── docs/
│   └── Контрольні_запитання_*.md # Lab Q&A documents
├── models/
│   ├── *.joblib                  # Trained model (DVC-tracked)
│   └── metrics.json              # Train/test metrics (DVC metrics)
├── artifacts/
│   └── <experiment>/
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── notebooks/
│   └── 01_eda.ipynb              # Exploratory data analysis
├── src/
│   ├── download_data.py          # Kaggle dataset downloader
│   ├── prepare.py                # Image → .npy preprocessing
│   ├── train.py                  # Model training + MLflow logging
│   ├── optimize.py               # Optuna HPO with MLflow nested runs
│   └── register_model.py         # MLflow Model Registry promotion
├── tests/
│   ├── test_data.py              # Pre-train: data validation (shape, NaN, range, balance)
│   ├── test_artifacts.py         # Post-train: artifact existence + Quality Gate
│   └── test_dag.py               # Airflow DAG structural tests (no live Airflow)
├── dvc.yaml                      # DVC pipeline (prepare + train stages)
├── dvc.lock                      # Reproducibility lock file
├── docker-compose.yml            # Airflow stack (webserver + scheduler + postgres)
├── Dockerfile                    # Multi-stage build for ML environment
├── pyproject.toml                # Dependencies + tool config (ruff, pytest, mypy)
├── Makefile                      # Developer shortcuts
└── uv.lock                       # Locked dependency tree
```

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Kaggle API credentials](https://www.kaggle.com/docs/api#authentication) — for data download
- Docker — for Airflow orchestration

```bash
make install        # install deps + pre-commit hooks
```

---

## Usage

### Data

```bash
make download-data  # download raw Chest X-Ray dataset from Kaggle
make prepare        # preprocess images → data/prepared/*.npy
```

### Training

```bash
make train          # run full DVC pipeline (prepare + train)
make mlflow         # launch MLflow UI at http://127.0.0.1:5000
```

DVC pipeline stages (`dvc.yaml`):

| Stage | Command | Outputs |
|-------|---------|---------|
| `prepare` | `python src/prepare.py` | `data/prepared/` |
| `train` | `python src/train.py` | `models/*.joblib`, `models/metrics.json`, `artifacts/` |

### Hyperparameter Optimisation (Optuna)

```bash
make optimize       # 20+ Optuna trials, logged as MLflow nested runs
```

### Testing

```bash
make test           # all tests (pre-train + post-train + DAG)
make test-data      # pre-train data validation only
make test-post      # post-train quality gate only
make test-dag       # Airflow DAG structural tests only
```

**Quality Gate thresholds** (enforced in CI):

| Metric | Threshold |
|--------|-----------|
| Test F1-score | ≥ 0.80 |
| Test Accuracy | ≥ 0.75 |
| Train–Test F1 gap | ≤ 0.15 |

### Linting

```bash
make lint           # ruff check
make format         # ruff format + autofix
```

---

## CI/CD Pipeline (GitHub Actions)

Triggered on every **push** and **pull request**. Steps:

1. **Lint** — `ruff check`
2. **DAG validation** — `pytest tests/test_dag.py`
3. **DVC pull** — authenticate via `DAGSHUB_TOKEN` secret, pull prepared data
4. **Pre-train tests** — `pytest tests/test_data.py`
5. **DVC repro** — run `prepare` + `train` pipeline
6. **Post-train / Quality Gate** — `pytest tests/test_artifacts.py`
7. **CML report** — posts metrics table + confusion matrix as a PR comment
8. **CD** — uploads `models/*.joblib` + `metrics.json` as a GitHub Actions artifact (on `main` push only)

### Required GitHub Secret

| Secret | Value |
|--------|-------|
| `DAGSHUB_TOKEN` | Your DAGsHub access token (used for DVC remote auth) |

---

## Data Versioning (DVC)

Remote storage: **DAGsHub S3-compatible bucket** (`s3://my-first-repo/dvc-store`).

```bash
make push-data      # push data + model artifacts to DVC remote
uv run dvc pull     # pull data (requires DAGSHUB_TOKEN in env or .dvc/config.local)
```

---

## Orchestration (Apache Airflow)

The DAG `ml_training_pipeline` runs `@weekly` to implement **Continuous Training**:

```
prepare_data → train_model → evaluate_model → register_model
```

- `evaluate_model` runs the pytest Quality Gate
- `register_model` compares the new F1 against the current MLflow `champion` alias and promotes if better

### Start Airflow locally

```bash
make airflow-init   # first-time DB migration + admin user creation
make airflow-up     # start webserver + scheduler
# Open http://localhost:8081  (admin / admin)
make airflow-down   # stop all containers
```

---

## Docker

Multi-stage `Dockerfile`:

- **Stage 1 (builder):** installs all deps with `uv` into an isolated venv
- **Stage 2 (runtime):** copies only the venv + `src/` + `conf/` — no build tools in the final image

```bash
docker build -t mlops-lab .
docker run --rm mlops-lab src/train.py --help
```

---

## Model Registry (MLflow)

`src/register_model.py` implements a promotion gate:

1. Reads `models/metrics.json` for the new model's `test_f1`
2. Fetches the current `champion` alias from MLflow Registry
3. Registers and promotes only if the new model is strictly better

```bash
uv run python src/register_model.py
```
