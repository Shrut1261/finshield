"""XGBoost fraud detection model with three class-imbalance strategies.

Benchmarks SMOTE, class_weight, and scale_pos_weight so the training
script can pick the best by AUC-PR on the validation set.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

BalancingStrategy = Literal["class_weight", "smote", "scale_pos_weight"]


class XGBoostFraudModel:
    """Production XGBoost model with configurable imbalance handling.

    Args:
        strategy: One of 'class_weight', 'smote', or 'scale_pos_weight'.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        random_state: Reproducibility seed.
    """

    DEFAULT_PARAMS: dict = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        strategy: BalancingStrategy = "class_weight",
        random_state: int = 42,
        **xgb_kwargs: object,
    ) -> None:
        self.strategy = strategy
        self.random_state = random_state
        self.params = {**self.DEFAULT_PARAMS, **xgb_kwargs}
        self.model: Optional[XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: list[str] = []

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.strategy == "smote":
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info("SMOTE: %d → %d samples", len(y), len(y_res))
            return X_res, y_res
        return X.values, y.values

    def _build_classifier(self, y: pd.Series) -> XGBClassifier:
        params = self.params.copy()
        if self.strategy == "class_weight":
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            logger.info("scale_pos_weight=%.2f (class_weight strategy)", params["scale_pos_weight"])
        elif self.strategy == "scale_pos_weight":
            params["scale_pos_weight"] = 100  # manual override for ultra-imbalanced
        return XGBClassifier(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "XGBoostFraudModel":
        """Train the model and build the SHAP explainer."""
        self.feature_names = list(X.columns)
        logger.info(
            "Training XGBoost (strategy=%s) on %d samples, fraud_rate=%.4f%%",
            self.strategy,
            len(y),
            y.mean() * 100,
        )

        X_train, y_train = self._prepare_data(X, y)
        self.model = self._build_classifier(pd.Series(y_train))

        fit_kwargs: dict = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [(eval_set[0].values, eval_set[1].values)]
            fit_kwargs["early_stopping_rounds"] = 50

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("XGBoost training complete. Best iteration: %s", getattr(self.model, "best_iteration", "N/A"))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X.values)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[list[dict[str, float]]]:
        """Return top-N SHAP feature contributions per transaction.

        Returns:
            List of lists: outer = transactions, inner = [{feature: shap_value}, ...]
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not built. Call fit() first.")

        shap_values = self.explainer.shap_values(X.values)
        results = []
        for i in range(len(X)):
            contributions = sorted(
                zip(self.feature_names, shap_values[i]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]
            results.append([{"feature": f, "shap_value": round(float(v), 6)} for f, v in contributions])
        return results

    def get_feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)
