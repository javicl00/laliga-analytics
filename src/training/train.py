"""Entrenamiento del modelo de prediccion (baseline + LightGBM).

Validacion walk-forward (sin data leakage temporal):
  - Train: J1  → J25
  - Val:   J26 → J30
  - Test:  J31 → J33  (held-out)

Metrica principal: RPS (Ranked Probability Score).
El modelo entrenado se guarda en models/lgbm_v1.pkl.

Features activas (10 columnas con datos reales):
  - ELO dinamico:     home_elo, away_elo, elo_diff
  - Forma reciente:   home/away_goals_for/against_last5
  - Contexto:         home_rest_days, away_rest_days, gameweek

Features pendientes (standings snapshot por jornada no disponible):
  home/away_points_total, home/away_table_position, position_diff,
  home/away_gd_total, home/away_pressure_index
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # Familia D: ELO dinamico
    "home_elo", "away_elo", "elo_diff",
    # Familia B: Forma reciente (ultimos 5 partidos)
    "home_goals_for_last5",   "home_goals_against_last5",
    "away_goals_for_last5",   "away_goals_against_last5",
    # Familia E: Contexto
    "home_rest_days", "away_rest_days",
    "gameweek",
]
TARGET_COL = "result"   # home | draw | away
CLASSES    = ["home", "draw", "away"]


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
        cum_pred = np.cumsum(probs[i])
        cum_true = np.cumsum(one_hot)
        total += np.sum((cum_pred - cum_true) ** 2) / 2
    return total / n


# ──────────────────────────────────────────────────────────
# Splits temporales
# ──────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_end_week: int = 25,
    val_end_week:   int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["gameweek_week"] <= train_end_week]
    val   = df[(df["gameweek_week"] > train_end_week) & (df["gameweek_week"] <= val_end_week)]
    test  = df[df["gameweek_week"] > val_end_week]
    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


# ──────────────────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────────────────

def baseline_probs(train: pd.DataFrame, n: int) -> np.ndarray:
    """Predice siempre la distribucion marginal del conjunto de entrenamiento."""
    counts = train[TARGET_COL].value_counts(normalize=True).to_dict()
    prob = [counts.get(c, 0.0) for c in CLASSES]
    return np.tile(prob, (n, 1))


# ──────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────

def train_lgbm(train: pd.DataFrame) -> LGBMClassifier:
    X = train[FEATURE_COLS].fillna(0)
    y = train[TARGET_COL]
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def train_logistic(train: pd.DataFrame) -> LogisticRegression:
    X = train[FEATURE_COLS].fillna(0)
    y = train[TARGET_COL]
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model


# ──────────────────────────────────────────────────────────
# Evaluacion
# ──────────────────────────────────────────────────────────

def evaluate(
    model, df: pd.DataFrame, split_name: str = "val"
) -> Dict[str, float]:
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].values
    probs = model.predict_proba(X)
    col_order = [list(model.classes_).index(c) for c in CLASSES]
    probs = probs[:, col_order]
    metrics = {
        "rps":      rps(y, probs),
        "log_loss": log_loss(y, probs, labels=CLASSES),
    }
    logger.info("%s metrics: %s", split_name, metrics)
    return metrics


# ──────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────

def run(
    features_df: pd.DataFrame,
    output_dir: str = "models",
    train_end_week: int = 25,
    val_end_week:   int = 30,
) -> Dict:
    # Solo eliminamos filas sin target; los NULLs en features los gestiona fillna(0)
    df = features_df.dropna(subset=[TARGET_COL])
    train, val, test = temporal_split(df, train_end_week, val_end_week)

    if train.empty or val.empty:
        raise ValueError(
            f"Split insuficiente: train={len(train)} val={len(val)}. "
            "Verifica que el JOIN matches+match_features devuelve filas con result IS NOT NULL "
            "y que gameweek_week esta poblado."
        )

    # Baseline
    base_probs = baseline_probs(train, len(val))
    base_rps   = rps(val[TARGET_COL].values, base_probs)
    logger.info("Baseline RPS (val): %.4f", base_rps)

    # Logistic regression
    lr = train_logistic(train)
    lr_metrics = evaluate(lr, val, "logistic_val")

    # LightGBM
    lgbm = train_lgbm(train)
    lgbm_metrics = evaluate(lgbm, val, "lgbm_val")

    # Test final solo con el mejor
    best_model = lgbm if lgbm_metrics["rps"] <= lr_metrics["rps"] else lr
    test_metrics = evaluate(best_model, test, "test") if not test.empty else {}

    # Persistir modelo
    Path(output_dir).mkdir(exist_ok=True)
    model_path = Path(output_dir) / "lgbm_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": FEATURE_COLS, "classes": CLASSES}, f)
    logger.info("Model saved to %s", model_path)

    return {
        "baseline_rps":  base_rps,
        "logistic":      lr_metrics,
        "lgbm":          lgbm_metrics,
        "test":          test_metrics,
        "model_path":    str(model_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sqlalchemy import create_engine, text as sqlt
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        matches = pd.read_sql(sqlt(
            "SELECT m.match_id, m.gameweek_week, m.result, "
            "f.home_elo, f.away_elo, f.elo_diff, "
            "f.home_goals_for_last5, f.home_goals_against_last5, "
            "f.away_goals_for_last5, f.away_goals_against_last5, "
            "f.home_rest_days, f.away_rest_days, f.gameweek "
            "FROM matches m "
            "JOIN match_features f USING (match_id) "
            "WHERE m.competition_main=TRUE AND m.result IS NOT NULL"
        ), conn)
    results = run(matches)
    print(results)
