"""Entrenamiento del modelo de prediccion (baseline + LightGBM).

Validacion walk-forward por temporada (sin data leakage temporal):
  - Train: todas las temporadas excepto las 2 ultimas
  - Val:   penultima temporada completa
  - Test:  ultima temporada (puede ser parcial si la actual esta en curso)

Metrica principal: RPS (Ranked Probability Score).
El modelo entrenado se guarda en models/lgbm_v1.pkl.

Features activas (18 columnas tras pruning):
  D - ELO:         elo_diff                         (home_elo/away_elo eliminados: redundantes)
  A - Standings:   home/away_points_total, home/away_table_position,
                   home/away_gd_total               (position_diff eliminado: redundante)
  B - Forma:       home/away_goals_for/against_last5
  E - Contexto:    home_rest_days, away_rest_days,
                   home_pressure_index, gameweek    (away_pressure_index eliminado: ruido)
  F - H2H:         h2h_home_wins, h2h_draws, h2h_away_wins
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # D: ELO (solo diferencial; home/away absolutos son redundantes con standings)
    "elo_diff",
    # A: Estado competitivo
    "home_points_total", "away_points_total",
    "home_table_position", "away_table_position",
    # position_diff eliminado: linealmente redundante con home/away_table_position
    "home_gd_total", "away_gd_total",
    # B: Forma reciente (ultimos 5, todos los campos)
    "home_goals_for_last5",   "home_goals_against_last5",
    "away_goals_for_last5",   "away_goals_against_last5",
    # E: Contexto
    "home_rest_days", "away_rest_days",
    "home_pressure_index",
    # away_pressure_index eliminado: coef 2.6%, ruido
    "gameweek",
    # F: Head-to-Head
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
]
TARGET_COL   = "result"   # home | draw | away
CLASSES      = ["home", "draw", "away"]
_CLASSES_LEX = sorted(CLASSES)   # ['away', 'draw', 'home']


# ──────────────────────────────────────────────────────────
# Metrica RPS
# ──────────────────────────────────────────────────────────

def rps(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Ranked Probability Score (menor es mejor)."""
    n = probs.shape[0]
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n):
        true_idx = CLASSES.index(y_true[i])
        one_hot  = np.zeros(3)
        one_hot[true_idx] = 1.0
        total += np.sum((np.cumsum(probs[i]) - np.cumsum(one_hot)) ** 2) / 2
    return total / n


def _reorder_probs(model, probs: np.ndarray) -> np.ndarray:
    classes: List[str] = list(model.classes_)
    return probs[:, [classes.index(c) for c in CLASSES]]


# ──────────────────────────────────────────────────────────
# Split walk-forward por temporada
# ──────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    val_season: Optional[int] = None,
    test_season: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk-forward split por temporada completa.

    Por defecto: test=ultima temporada, val=penultima, train=el resto.
    """
    seasons = sorted(df["season_id"].dropna().unique().tolist())
    logger.info("Temporadas disponibles: %s", seasons)

    if len(seasons) < 2:
        raise ValueError(f"Se necesitan al menos 2 temporadas, hay {len(seasons)}")

    if test_season is None:
        test_season = seasons[-1]
    if val_season is None:
        val_season = seasons[-2]

    train_seasons = [s for s in seasons if s not in (val_season, test_season)]
    train = df[df["season_id"].isin(train_seasons)]
    val   = df[df["season_id"] == val_season]
    test  = df[df["season_id"] == test_season]

    logger.info(
        "Split: train=%d (T:%s) | val=%d (T:%s) | test=%d (T:%s)",
        len(train), train_seasons, len(val), val_season, len(test), test_season,
    )
    return train, val, test


# ──────────────────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────────────────

def baseline_probs(train: pd.DataFrame, n: int) -> np.ndarray:
    counts = train[TARGET_COL].value_counts(normalize=True).to_dict()
    return np.tile([counts.get(c, 0.0) for c in CLASSES], (n, 1))


# ──────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────

def train_lgbm(train: pd.DataFrame) -> LGBMClassifier:
    X = train[FEATURE_COLS].fillna(0)
    y = train[TARGET_COL]
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def train_logistic(train: pd.DataFrame) -> Pipeline:
    X = train[FEATURE_COLS].fillna(0)
    y = train[TARGET_COL]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]).fit(X, y)


# ──────────────────────────────────────────────────────────
# Evaluacion
# ──────────────────────────────────────────────────────────

def evaluate(model, df: pd.DataFrame, split_name: str = "val") -> Dict[str, float]:
    X         = df[FEATURE_COLS].fillna(0)
    y         = df[TARGET_COL].values
    probs     = _reorder_probs(model, model.predict_proba(X))
    probs_lex = probs[:, [CLASSES.index(c) for c in _CLASSES_LEX]]
    metrics   = {
        "rps":      rps(y, probs),
        "log_loss": log_loss(y, probs_lex, labels=_CLASSES_LEX),
    }
    logger.info("%s metrics: %s", split_name, metrics)
    return metrics


# ──────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────

def run(
    features_df: pd.DataFrame,
    output_dir: str = "models",
    val_season: Optional[int] = None,
    test_season: Optional[int] = None,
) -> Dict:
    df = features_df.dropna(subset=[TARGET_COL])
    train, val, test = temporal_split(df, val_season, test_season)

    if train.empty or val.empty:
        raise ValueError(
            f"Split insuficiente: train={len(train)} val={len(val)}."
        )

    base_rps = rps(val[TARGET_COL].values, baseline_probs(train, len(val)))
    logger.info("Baseline RPS (val): %.4f", base_rps)

    lr           = train_logistic(train)
    lr_metrics   = evaluate(lr, val, "logistic_val")
    lgbm         = train_lgbm(train)
    lgbm_metrics = evaluate(lgbm, val, "lgbm_val")

    best_model   = lgbm if lgbm_metrics["rps"] <= lr_metrics["rps"] else lr
    test_metrics = evaluate(best_model, test, "test") if not test.empty else {}

    Path(output_dir).mkdir(exist_ok=True)
    model_path = Path(output_dir) / "lgbm_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        best_model,
            "feature_cols": FEATURE_COLS,
            "classes":      CLASSES,
        }, f)
    logger.info("Model saved to %s", model_path)

    return {
        "baseline_rps": base_rps,
        "logistic":     lr_metrics,
        "lgbm":         lgbm_metrics,
        "test":         test_metrics,
        "model_path":   str(model_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sqlalchemy import create_engine, text as sqlt
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        matches = pd.read_sql(sqlt(
            "SELECT m.match_id, m.season_id, m.gameweek_week, m.result, "
            "f.elo_diff, "
            "f.home_points_total, f.away_points_total, "
            "f.home_table_position, f.away_table_position, "
            "f.home_gd_total, f.away_gd_total, "
            "f.home_goals_for_last5, f.home_goals_against_last5, "
            "f.away_goals_for_last5, f.away_goals_against_last5, "
            "f.home_rest_days, f.away_rest_days, "
            "f.home_pressure_index, "
            "f.gameweek, f.h2h_home_wins, f.h2h_draws, f.h2h_away_wins "
            "FROM matches m "
            "JOIN match_features f USING (match_id) "
            "WHERE m.competition_main=TRUE AND m.result IS NOT NULL"
        ), conn)
    results = run(matches)
    print(results)
