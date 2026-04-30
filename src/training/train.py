"""Entrenamiento del modelo de prediccion (baseline + LightGBM).

Validacion walk-forward (sin data leakage temporal):
  - Train: J1  → J25
  - Val:   J26 → J30
  - Test:  J31 → J33  (held-out)

Metrica principal: RPS (Ranked Probability Score).
El modelo entrenado se guarda en models/lgbm_v1.pkl.

Features v2 (19 columnas, alineadas con build_features.FEATURE_COLUMNS):
  - ELO dinamico: home_elo, away_elo, elo_diff
  - Estado competitivo: home/away_points_total, home/away_table_position,
    position_diff, home/away_gd_total
  - Forma reciente (ultimos 5): home/away_goals_for/against_last5
  - Contexto: gameweek, home/away_rest_days, home/away_pressure_index
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # Familia D: ELO
    "home_elo", "away_elo", "elo_diff",
    # Familia A: Estado competitivo
    "home_points_total", "away_points_total",
    "home_table_position", "away_table_position", "position_diff",
    "home_gd_total", "away_gd_total",
    # Familia B: Forma reciente
    "home_goals_for_last5", "home_goals_against_last5",
    "away_goals_for_last5", "away_goals_against_last5",
    # Familia E: Contexto
    "home_rest_days", "away_rest_days",
    "home_pressure_index", "away_pressure_index",
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
    df = features_df.dropna(subset=[TARGET_COL] + FEATURE_COLS)
    train, val, test = temporal_split(df, train_end_week, val_end_week)

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
            "SELECT m.match_id, m.gameweek_week, m.result, f.* "
            "FROM matches m "
            "JOIN match_features f USING (match_id) "
            "WHERE m.competition_main=TRUE AND m.result IS NOT NULL"
        ), conn)
    results = run(matches)
    print(results)
