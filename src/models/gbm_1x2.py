"""Modelos GBM (LightGBM / CatBoost) para clasificación 1X2.

Ambos modelos siguen la misma interfaz: fit + predict_proba.
Se pueden combinar en el stacker de calibración.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LABEL_ENCODER = {"H": 0, "D": 1, "A": 2}
LABEL_DECODER = {0: "H", 1: "D", 2: "A"}


def encode_labels(y) -> np.ndarray:
    return np.array([LABEL_ENCODER[v] for v in y])


class LGBMModel:
    """Clasificador LightGBM para 1X2."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        colsample_bytree: float = 0.8,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            random_state=random_state,
            class_weight="balanced",
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMModel":
        y_enc = encode_labels(y) if y.dtype == object else y
        self.model.fit(X, y_enc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class CatBoostModel:
    """Clasificador CatBoost para 1X2."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        random_seed: int = 42,
    ) -> None:
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0,
            eval_metric="MultiClass",
            loss_function="MultiClass",
            auto_class_weights="Balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostModel":
        y_enc = encode_labels(y) if y.dtype == object else y
        self.model.fit(X, y_enc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
