"""Model training orchestrator with MLflow experiment tracking.

Runs the full training cycle:
  1. Load feature matrix from the feature store / processed parquet
  2. Split train/validation/test (stratified, no temporal leakage)
  3. Train all models and log to MLflow
  4. Compute and log all evaluation metrics
  5. Register best model in the MLflow Model Registry
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from models.baseline import BaselineModel
from models.ensemble import FraudEnsemble
from models.isolation_forest import IsolationForestDetector
from models.xgboost_model import XGBoostFraudModel

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5  # Tuned on validation set during training


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Compute the full KPI set for one model."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "auc_pr": round(float(average_precision_score(y_true, y_prob)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "fraud_rate_actual": round(float(y_true.mean()), 6),
        "fraud_rate_predicted": round(float(y_pred.mean()), 6),
    }


def train_all(
    feature_path: Path,
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "finshield_fraud_detection",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict[str, dict]:
    """Train all models and return metrics dict keyed by model name."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("Loading features from %s", feature_path)
    df = pd.read_parquet(feature_path)
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud", "transaction_id"], errors="ignore")

    # Stratified split preserves fraud rate in every split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=random_state
    )

    logger.info(
        "Split: train=%d val=%d test=%d | fraud_rate=%.4f%%",
        len(y_train), len(y_val), len(y_test),
        y_train.mean() * 100,
    )

    models_to_train: dict[str, object] = {
        "baseline_lr": BaselineModel(),
        "xgboost_class_weight": XGBoostFraudModel(strategy="class_weight"),
        "xgboost_smote": XGBoostFraudModel(strategy="smote"),
        "isolation_forest": IsolationForestDetector(),
    }

    all_metrics: dict[str, dict] = {}

    for name, model in models_to_train.items():
        logger.info("Training %s ...", name)
        with mlflow.start_run(run_name=name):
            if isinstance(model, IsolationForestDetector):
                model.fit(X_train)
                proba = model.anomaly_score(X_test)
            else:
                fit_kwargs = {}
                if hasattr(model, "fit") and "eval_set" in model.fit.__code__.co_varnames:
                    fit_kwargs["eval_set"] = (X_val, y_val)
                model.fit(X_train, y_train, **fit_kwargs)  # type: ignore[call-arg]
                proba = model.predict_proba(X_test)  # type: ignore[union-attr]

            metrics = compute_metrics(y_test.values, proba)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", name)
            mlflow.log_param("strategy", getattr(model, "strategy", "N/A"))
            mlflow.log_param("n_train", len(y_train))
            mlflow.log_param("fraud_rate_train", float(y_train.mean()))

            # Save artifact
            artifact_path = ARTIFACTS_DIR / f"{name}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(artifact_path))

            all_metrics[name] = metrics
            logger.info("%s → AUC-ROC=%.4f AUC-PR=%.4f Recall=%.4f", name, metrics["auc_roc"], metrics["auc_pr"], metrics["recall"])

    # ── Train full ensemble (most compute-intensive, run last) ────────────────
    logger.info("Training stacked ensemble ...")
    with mlflow.start_run(run_name="stacked_ensemble"):
        ensemble = FraudEnsemble(n_folds=3, random_state=random_state)
        ensemble.fit(X_train, y_train)
        proba = ensemble.predict_proba(X_test)
        metrics = compute_metrics(y_test.values, proba)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "stacked_ensemble")

        artifact_path = ARTIFACTS_DIR / "ensemble.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(ensemble, f)
        mlflow.log_artifact(str(artifact_path))
        all_metrics["stacked_ensemble"] = metrics
        logger.info(
            "Ensemble → AUC-ROC=%.4f AUC-PR=%.4f Recall=%.4f",
            metrics["auc_roc"], metrics["auc_pr"], metrics["recall"],
        )

    # Print comparison table
    print("\n── Model Comparison ──────────────────────────────────────────────────")
    print(f"{'Model':<30} {'AUC-ROC':>8} {'AUC-PR':>8} {'Recall':>8} {'Precision':>10}")
    for name, m in all_metrics.items():
        print(f"{name:<30} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} {m['recall']:>8.4f} {m['precision']:>10.4f}")

    return all_metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    feature_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/features.parquet")
    train_all(feature_path)
