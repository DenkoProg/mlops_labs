
.PHONY: install
install: ## Install dependencies and setup pre-commit hooks
	@echo "🚀 Installing dependencies from lockfile"
	@uv sync --frozen
	@uv run pre-commit install

.PHONY: lint
lint: ## Run ruff linter
	uv run ruff check

.PHONY: format
format: ## Format code and fix linting issues
	uv run ruff format
	uv run ruff check --fix

.PHONY: download-data
download-data: ## Download Chest X-Ray dataset from Kaggle
	uv run python src/download_data.py

.PHONY: prepare
prepare: ## Run data preparation stage (dvc)
	uv run python src/prepare.py data/raw/chest_xray data/prepared --img-size 64

.PHONY: train
train: ## Run full DVC pipeline (prepare + train)
	uv run dvc repro

.PHONY: push-data
push-data: ## Push data and model artifacts to DVC remote
	uv run dvc push

.PHONY: optimize
optimize: ## Run Optuna hyperparameter optimization (20+ trials, MLflow nested runs)
	uv run python src/optimize.py

.PHONY: mlflow
mlflow: ## Launch MLflow UI at http://127.0.0.1:5000
	uv run mlflow ui

.PHONY: eda
eda: ## Open EDA notebook
	uv run jupyter lab notebooks/01_eda.ipynb

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help