"""Entrenamiento del modelo de prediccion (baseline + LightGBM).

Validacion walk-forward por temporada (sin data leakage temporal):
  - Train: todas las temporadas excepto las 2 ultimas
  - Val:   penultima temporada completa
  - Test:  ultima temporada (puede ser parcial si la actual esta en curso)

Metrica principal: RPS (Ranked Probability Score).
El modelo entrenado se guarda en models/lgbm_v1.pkl.

Seleccion de modelo:
  Se comparan LightGBM y LogisticRegression por RPS en val. Gana el de
  menor RPS. La comparacion es valida porque ambos usan el mismo conjunto
  de features (determinado por cobertura en train, ver _available()).

Features activas:
  D - ELO:         home_elo, away_elo  (diferencial implicito; nivel absoluto)
  A - Standings:   home/away_points_total, home/away_table_position,
                   home/away_gd_total
  B - Forma:       home/away_goals_for/against_last5
  E - Contexto:    home_rest_days, away_rest_days,
                   home_pressure_index, gameweek
  F - H2H:         h2h_home_wins, h2h_draws, h2h_away_wins
  G - Opta stats:  home/away_possession_last5, home/away_ppda_last5,
                   home/away_shots_ot_last5, home/away_bigchances_last5
                   SOLO incluidas si cobertura en train >= OPTA_THRESHOLD.
                   NaN nativo para LightGBM; fillna(0) para LogisticRegression.
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

# Umbral minimo de cobertura (filas no-NaN / total) para incluir familia G
OPTA_THRESHOLD = 0.10

BASE_COLS = [
    # D: ELO — absolutos por equipo (el diferencial queda implicito)
    "home_elo", "away_elo",
    # A: Estado competitivo
    "home_points_total", "away_points_total",
    "home_table_position", "away_table_position",
    "home_gd_total", "away_gd_total",
    # B: Forma reciente
    "home_goals_for_last5",   "home_goals_against_last5",
    "away_goals_for_last5",   "away_goals_against_last5",
    # E: Contexto
    "home_rest_days", "away_rest_days",
    "home_pressure_index",
    "gameweek",
    # F: Head-to-Head
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
]

OPTA_COLS = [
    "home_possession_last5", "away_possession_last5",
    "home_ppda_last5",       "away_ppda_last5",
    "home_shots_ot_last5",   "away_shots_ot_last5",
    "home_bigchances_last5", "away_bigchances_last5",
]

FEATURE_COLS = BASE_COLS + OPTA_COLS   # lista completa para referencia
TARGET_COL   = "result"
CLASSES      = ["home", "draw", "away"]
_CLASSES_LEX = sorted(CLASSES)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _available(df: pd.DataFrame) -> List[str]:
    """Devuelve columnas utiles para el modelo dado un DataFrame.

    Incluye familia G (Opta) solo si su cobertura en df supera OPTA_THRESHOLD.
    Esto evita que LightGBM aprenda splits espurios sobre columnas 100% NaN
    cuando ningun partido del split tiene datos Opta.
    """
    base = [c for c in BASE_COLS if c in df.columns]

    opta_present = [c for c in OPTA_COLS if c in df.columns]
    if opta_present:
        coverage = df[opta_present].notna().mean().mean()
    else:
        coverage = 0.0

    if coverage >= OPTA_THRESHOLD:
        logger.info(
            "Features Opta INCLUIDAS (cobertura=%.1f%% >= %.0f%%): %s",
            coverage * 100, OPTA_THRESHOLD * 100, opta_present,
        )
        return base + opta_present
    else:
        logger.info(
            "Features Opta EXCLUIDAS (cobertura=%.1f%% < %.0f%%) — "
            "se activaran cuando train tenga suficientes temporadas con opta_id.",
            coverage * 100, OPTA_THRESHOLD * 100,
        )
        return base


def _X_lgbm(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Features para LightGBM: NaN nativo (sin fillna)."""
    return df[cols]


def _X_sklearn(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Features para sklearn: fillna(0) porque Pipeline no tolera NaN."""
    return df[cols].fillna(0)


# ────────────────────────────────────────────────────────────
# Metrica RPS
# ────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────
# Split walk-forward por temporada
# ────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    val_season: Optional[int] = None,
    test_season: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


# ────────────────────────────────────────────────────────────
# Baseline
# ────────────────────────────────────────────────────────────

def baseline_probs(train: pd.DataFrame, n: int) -> np.ndarray:
    counts = train[TARGET_COL].value_counts(normalize=True).to_dict()
    return np.tile([counts.get(c, 0.0) for c in CLASSES], (n, 1))


# ────────────────────────────────────────────────────────────
# Entrenamiento
# ────────────────────────────────────────────────────────────

def train_lgbm(train: pd.DataFrame) -> LGBMClassifier:
    cols = _available(train)
    X = _X_lgbm(train, cols)
    y = train[TARGET_COL]
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    model._feature_cols_used = cols
    return model


def train_logistic(train: pd.DataFrame) -> Pipeline:
    cols = _available(train)
    X = _X_sklearn(train, cols)
    y = train[TARGET_COL]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]).fit(X, y)
    pipe._feature_cols_used = cols
    return pipe


# ────────────────────────────────────────────────────────────
# Evaluacion
# ────────────────────────────────────────────────────────────

def evaluate(model, df: pd.DataFrame, split_name: str = "val") -> Dict[str, float]:
    cols    = [c for c in getattr(model, "_feature_cols_used", FEATURE_COLS) if c in df.columns]
    is_lgbm = isinstance(model, LGBMClassifier)
    X       = _X_lgbm(df, cols) if is_lgbm else _X_sklearn(df, cols)
    y       = df[TARGET_COL].values
    probs     = _reorder_probs(model, model.predict_proba(X))
    probs_lex = probs[:, [CLASSES.index(c) for c in _CLASSES_LEX]]
    metrics   = {
        "rps":      rps(y, probs),
        "log_loss": log_loss(y, probs_lex, labels=_CLASSES_LEX),
    }
    logger.info("%s metrics: %s", split_name, metrics)
    return metrics


# ────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────

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

    base_rps     = rps(val[TARGET_COL].values, baseline_probs(train, len(val)))
    logger.info("Baseline RPS (val): %.4f", base_rps)

    lr           = train_logistic(train)
    lr_metrics   = evaluate(lr, val, "logistic_val")
    lgbm         = train_lgbm(train)
    lgbm_metrics = evaluate(lgbm, val, "lgbm_val")

    # Seleccion dinamica: gana el modelo con menor RPS en val.
    # La comparacion es justa porque _available() usa el mismo conjunto
    # de features para ambos (Opta excluida si cobertura < OPTA_THRESHOLD).
    if lgbm_metrics["rps"] <= lr_metrics["rps"]:
        best_model  = lgbm
        winner_name = "LightGBM"
    else:
        best_model  = lr
        winner_name = "LogisticRegression"
    logger.info(
        "Modelo seleccionado: %s (LGBM RPS=%.4f | LR RPS=%.4f)",
        winner_name, lgbm_metrics["rps"], lr_metrics["rps"],
    )

    test_metrics = evaluate(best_model, test, "test") if not test.empty else {}

    Path(output_dir).mkdir(exist_ok=True)
    model_path = Path(output_dir) / "lgbm_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        best_model,
            "feature_cols": getattr(best_model, "_feature_cols_used", FEATURE_COLS),
            "classes":      CLASSES,
        }, f)
    logger.info("Model saved to %s", model_path)

    return {
        "baseline_rps": base_rps,
        "logistic":     lr_metrics,
        "lgbm":         lgbm_metrics,
        "winner":       winner_name,
        "test":         test_metrics,
        "model_path":   str(model_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from sqlalchemy import create_engine, text as sqlt
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        matches = pd.read_sql(sqlt("""
            SELECT
                m.match_id, m.season_id, m.result,
                f.home_elo, f.away_elo,
                f.home_points_total,    f.away_points_total,
                f.home_table_position,  f.away_table_position,
                f.home_gd_total,        f.away_gd_total,
                f.home_goals_for_last5, f.home_goals_against_last5,
                f.away_goals_for_last5, f.away_goals_against_last5,
                f.home_rest_days,       f.away_rest_days,
                f.home_pressure_index,
                f.gameweek,
                f.h2h_home_wins, f.h2h_draws, f.h2h_away_wins,
                f.home_possession_last5, f.away_possession_last5,
                f.home_ppda_last5,       f.away_ppda_last5,
                f.home_shots_ot_last5,   f.away_shots_ot_last5,
                f.home_bigchances_last5, f.away_bigchances_last5
            FROM matches m
            JOIN match_features f USING (match_id)
            WHERE m.competition_main = TRUE
              AND m.result IS NOT NULL
        """), conn)
    results = run(matches)
    print(results)
