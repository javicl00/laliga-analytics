"""Modelo baseline de goles basado en regresión de Poisson.

Entrena dos modelos independientes para home_goals y away_goals.
Devuelve lambdas de Poisson y probabilidades para scorelines y 1X2.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor


class PoissonBaseline:
    """Regresión de Poisson para predicción de goles home y away.

    Parameters
    ----------
    alpha:
        Regularización L2 de la regresión de Poisson.
    max_iter:
        Iteraciones máximas del solver.
    max_goals:
        Máximo de goles considerado al calcular distribución de scorelines.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 500,
        max_goals: int = 8,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_goals = max_goals
        self._home_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self._away_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
    ) -> "PoissonBaseline":
        self._home_model.fit(X, y_home)
        self._away_model.fit(X, y_away)
        self._fitted = True
        return self

    def predict_goals(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (lambda_home, lambda_away) para cada muestra."""
        if not self._fitted:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")
        return self._home_model.predict(X), self._away_model.predict(X)

    def predict_proba_1x2(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Devuelve array (N, 3) con probabilidades [home, draw, away]."""
        lam_home, lam_away = self.predict_goals(X)
        probs = np.zeros((len(X), 3))
        for i, (lh, la) in enumerate(zip(lam_home, lam_away)):
            matrix = self._score_matrix(lh, la)
            probs[i, 0] = np.sum(np.tril(matrix, -1))   # home win
            probs[i, 1] = np.sum(np.diag(matrix))        # draw
            probs[i, 2] = np.sum(np.triu(matrix, 1))     # away win
        return probs

    def predict_top_scorelines(
        self,
        X: np.ndarray,
        top_n: int = 5,
    ) -> List[List[Dict]]:
        """Devuelve los top_n scorelines más probables por muestra."""
        lam_home, lam_away = self.predict_goals(X)
        results = []
        for lh, la in zip(lam_home, lam_away):
            matrix = self._score_matrix(lh, la)
            scorelines = [
                {"home": i, "away": j, "prob": matrix[i, j]}
                for i, j in itertools.product(range(self.max_goals + 1), repeat=2)
            ]
            scorelines.sort(key=lambda x: x["prob"], reverse=True)
            results.append(scorelines[:top_n])
        return results

    def _score_matrix(
        self,
        lam_home: float,
        lam_away: float,
    ) -> np.ndarray:
        """Matriz de probabilidades de scorelines hasta max_goals."""
        mg = self.max_goals
        ph = poisson.pmf(np.arange(mg + 1), lam_home)
        pa = poisson.pmf(np.arange(mg + 1), lam_away)
        return np.outer(ph, pa)
