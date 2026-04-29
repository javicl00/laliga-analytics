"""Validación walk-forward temporal para modelos de fútbol.

Toda la validación es estrictamente temporal:
  - Nunca se mezclan datos de futuro con pasado en el entrenamiento.
  - Los splits se definen por jornada o por temporada completa.
  - Se calculan métricas por bloque y agregadas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.models.calibration import brier_score_multiclass, ranked_probability_score
from src.models.gbm_1x2 import encode_labels

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    split_index: int
    train_size: int
    val_size: int
    log_loss: float
    brier_score: float
    rps: float
    accuracy: float
    probs: np.ndarray = field(repr=False)
    y_true: np.ndarray = field(repr=False)


def gameweek_splits(
    df: pd.DataFrame,
    min_train_gameweeks: int = 10,
    val_gameweeks: int = 1,
    step: int = 1,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Genera pares (train, val) avanzando jornada a jornada."""
    gws = sorted(df["gameweek"].unique())
    for i in range(min_train_gameweeks, len(gws) - val_gameweeks + 1, step):
        train_gws = gws[:i]
        val_gws = gws[i: i + val_gameweeks]
        train = df[df["gameweek"].isin(train_gws)]
        val = df[df["gameweek"].isin(val_gws)]
        yield train, val


def season_splits(
    df: pd.DataFrame,
    min_train_seasons: int = 1,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Genera pares (train, val) usando temporadas completas como bloques."""
    seasons = sorted(df["season_id"].unique())
    for i in range(min_train_seasons, len(seasons)):
        train = df[df["season_id"].isin(seasons[:i])]
        val = df[df["season_id"] == seasons[i]]
        yield train, val


class WalkForwardEvaluator:
    """Evalúa un modelo con validación walk-forward temporal.

    Parameters
    ----------
    model_factory:
        Callable sin argumentos que devuelve una instancia nueva del modelo.
    feature_cols:
        Columnas de features a usar.
    target_col:
        Columna objetivo (string H/D/A).
    min_train_gameweeks:
        Jornadas mínimas de entrenamiento.
    val_gameweeks:
        Jornadas de validación por bloque.
    """

    def __init__(
        self,
        model_factory,
        feature_cols: List[str],
        target_col: str = "y_1x2",
        min_train_gameweeks: int = 10,
        val_gameweeks: int = 1,
    ) -> None:
        self.model_factory = model_factory
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.min_train_gw = min_train_gameweeks
        self.val_gw = val_gameweeks

    def run(
        self,
        features_df: pd.DataFrame,
    ) -> List[SplitResult]:
        """Ejecuta walk-forward y devuelve lista de SplitResult."""
        results: List[SplitResult] = []
        for idx, (train, val) in enumerate(
            gameweek_splits(features_df, self.min_train_gw, self.val_gw)
        ):
            X_train = train[self.feature_cols].fillna(0).values
            y_train_raw = train[self.target_col].values
            y_train = encode_labels(y_train_raw)

            X_val = val[self.feature_cols].fillna(0).values
            y_val_raw = val[self.target_col].values
            y_val = encode_labels(y_val_raw)

            model = self.model_factory()
            model.fit(X_train, y_train_raw)
            probs = model.predict_proba(X_val)

            result = SplitResult(
                split_index=idx,
                train_size=len(X_train),
                val_size=len(X_val),
                log_loss=float(log_loss(y_val, probs)),
                brier_score=brier_score_multiclass(y_val, probs),
                rps=ranked_probability_score(y_val, probs),
                accuracy=float(np.mean(np.argmax(probs, axis=1) == y_val)),
                probs=probs,
                y_true=y_val,
            )
            logger.info(
                "Split %d | train=%d val=%d | ll=%.4f brier=%.4f rps=%.4f acc=%.4f",
                idx, result.train_size, result.val_size,
                result.log_loss, result.brier_score, result.rps, result.accuracy,
            )
            results.append(result)
        return results

    def summary(self, results: List[SplitResult]) -> pd.DataFrame:
        return pd.DataFrame([{
            "split": r.split_index,
            "train_size": r.train_size,
            "val_size": r.val_size,
            "log_loss": r.log_loss,
            "brier_score": r.brier_score,
            "rps": r.rps,
            "accuracy": r.accuracy,
        } for r in results])
