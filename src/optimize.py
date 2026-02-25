import logging
from pathlib import Path

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from omegaconf import DictConfig
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_prepared_data(prepared_dir: Path) -> tuple:
    X_train = np.load(prepared_dir / "X_train.npy")
    y_train = np.load(prepared_dir / "y_train.npy")
    X_test = np.load(prepared_dir / "X_test.npy")
    y_test = np.load(prepared_dir / "y_test.npy")
    return X_train, y_train, X_test, y_test


def compute_metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
    }


def make_objective(X_train, y_train, X_test, y_test, cfg: DictConfig, parent_run_id: str):
    model_cfg = cfg.model

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", model_cfg.n_estimators.low, model_cfg.n_estimators.high)
        max_depth = trial.suggest_int("max_depth", model_cfg.max_depth.low, model_cfg.max_depth.high)
        min_samples_split = trial.suggest_int(
            "min_samples_split",
            model_cfg.min_samples_split.low,
            model_cfg.min_samples_split.high,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=model_cfg.class_weight,
            random_state=model_cfg.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)

        # Log each trial as a nested (child) MLflow run
        with mlflow.start_run(
            run_name=f"trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_params(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "trial_number": trial.number,
                }
            )
            train_metrics = compute_metrics(y_train, model.predict(X_train), "train")
            test_metrics = compute_metrics(y_test, y_pred, "test")
            mlflow.log_metrics({**train_metrics, **test_metrics})
            mlflow.set_tag("optuna_trial", trial.number)

        return test_f1

    return objective


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    prepared_dir = Path(hydra.utils.get_original_cwd()) / cfg.data.prepared_dir
    models_dir = Path(hydra.utils.get_original_cwd()) / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    log.info(f"Loading prepared data from {prepared_dir}")
    X_train, y_train, X_test, y_test = load_prepared_data(prepared_dir)
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_tracking_uri(f"sqlite:///{Path(hydra.utils.get_original_cwd()) / 'mlruns' / 'mlflow.db'}")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="hpo_study") as parent_run:
        mlflow.set_tag("optuna_n_trials", cfg.optuna.n_trials)
        mlflow.set_tag("optuna_direction", cfg.optuna.direction)
        mlflow.set_tag("optuna_seed", cfg.optuna.seed)

        sampler = optuna.samplers.TPESampler(seed=cfg.optuna.seed)
        study = optuna.create_study(
            direction=cfg.optuna.direction,
            sampler=sampler,
            study_name=cfg.mlflow.experiment_name,
        )

        objective = make_objective(X_train, y_train, X_test, y_test, cfg, parent_run.info.run_id)
        study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)

        best = study.best_trial
        log.info(f"Best trial #{best.number}: test_f1={best.value:.4f}")
        log.info(f"Best params: {best.params}")

        # Log best results on the parent run
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_test_f1", best.value)
        mlflow.set_tag("best_trial_number", best.number)

        # Retrain final model with best hyperparameters
        log.info("Retraining final model with best params...")
        final_model = RandomForestClassifier(
            n_estimators=best.params["n_estimators"],
            max_depth=best.params["max_depth"],
            min_samples_split=best.params["min_samples_split"],
            class_weight=cfg.model.class_weight,
            random_state=cfg.model.random_state,
            n_jobs=-1,
        )
        final_model.fit(X_train, y_train)

        y_train_pred = final_model.predict(X_train)
        y_test_pred = final_model.predict(X_test)

        final_train_metrics = compute_metrics(y_train, y_train_pred, "final_train")
        final_test_metrics = compute_metrics(y_test, y_test_pred, "final_test")
        mlflow.log_metrics({**final_train_metrics, **final_test_metrics})

        log.info("--- Final Train Metrics ---")
        for k, v in final_train_metrics.items():
            log.info(f"  {k}: {v:.4f}")
        log.info("--- Final Test Metrics ---")
        for k, v in final_test_metrics.items():
            log.info(f"  {k}: {v:.4f}")

        # Save model
        mlflow.sklearn.log_model(final_model, "best_rf_model")
        model_path = models_dir / "best_rf.joblib"
        joblib.dump(final_model, model_path)
        log.info(f"Final model saved to {model_path}")


if __name__ == "__main__":
    main()
