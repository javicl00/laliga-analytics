"""Calibración probabilística para modelos 1X2.

Soporta isotonic regression, Platt scaling y calibración
de conjunto (stacking) de varios modelos base.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Calibrador isotónico por clase para arrays de probabilidades.

    Entrena un regresor isotónico por cada clase (0=H, 1=D, 2=A)
    y renormaliza para que las probabilidades sumen 1.
    """

    def __init__(self) -> None:
        self._calibrators = [IsotonicRegression(out_of_bounds="clip") for _ in range(3)]
        self._fitted = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """probs: (N,3), y: etiquetas enteras 0/1/2."""
        for cls in range(3):
            binary = (y == cls).astype(float)
            self._calibrators[cls].fit(probs[:, cls], binary)
        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrador no entrenado.")
        calibrated = np.stack(
            [self._calibrators[cls].transform(probs[:, cls]) for cls in range(3)],
            axis=1,
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.where(row_sums == 0, 1, row_sums)


class StackedEnsemble:
    """Combina probabilidades de N modelos base mediante stacking logístico.

    Parameters
    ----------
    base_models:
        Lista de modelos con interfaz predict_proba(X) -> (N, 3).
    meta_learning_rate:
        Regularización del meta-modelo logístico.
    """

    def __init__(self, base_models: List, meta_learning_rate: float = 1.0) -> None:
        self.base_models = base_models
        self._meta = LogisticRegression(
            C=meta_learning_rate,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500,
        )
        self._calibrator = IsotonicCalibrator()
        self._fitted = False

    def fit(
        self,
        X_meta: np.ndarray,
        y_meta: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> "StackedEnsemble":
        """Entrena meta-modelo con X_meta e isotonic calibration con X_calib."""
        meta_input = self._stack(X_meta)
        self._meta.fit(meta_input, y_meta)
        raw_probs = self._meta.predict_proba(self._stack(X_calib))
        self._calibrator.fit(raw_probs, y_calib)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Ensemble no entrenado.")
        raw = self._meta.predict_proba(self._stack(X))
        return self._calibrator.transform(raw)

    def _stack(self, X: np.ndarray) -> np.ndarray:
        parts = [m.predict_proba(X) for m in self.base_models]
        return np.hstack(parts)


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Brier score multiclase. Menor es mejor."""
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Ranked Probability Score para 1X2. Menor es mejor."""
    n = len(y_true)
    n_classes = probs.shape[1]
    cum_probs = np.cumsum(probs, axis=1)
    one_hot = np.eye(n_classes)[y_true]
    cum_true = np.cumsum(one_hot, axis=1)
    return float(np.mean(np.sum((cum_probs - cum_true) ** 2, axis=1) / (n_classes - 1)))
