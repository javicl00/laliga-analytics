"""Familia G: features rolling de stats Opta por partido (prepartido, sin leakage).

Requiere que match_stats_opta este poblada por fetch_match_stats.py.

Por cada partido calcula el rolling avg de los ultimos N partidos
(home o away, todos los campos) ANTERIORES al kickoff del partido objetivo.

Features generadas (8 columnas):
  home/away_possession_last5  -- % posesion media
  home/away_ppda_last5        -- PPDA medio (menor = mas presion)
  home/away_shots_ot_last5    -- tiros a puerta medios
  home/away_bigchances_last5  -- grandes ocasiones creadas medias

Ejecucion independiente:
  docker compose run --rm etl python -m src.features.opta_features
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLLING_N = 5   # ventana de partidos para rolling avg
_OPTA_COLS = ["possession_pct", "ppda", "shots_on_target", "big_chances_created"]


def _load_data(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga matches y match_stats_opta."""
    with engine.connect() as conn:
        matches = pd.read_sql(text(
            "SELECT match_id, season_id, kickoff_at, home_team_id, away_team_id "
            "FROM matches WHERE competition_main = TRUE ORDER BY kickoff_at"
        ), conn)
        stats = pd.read_sql(text(
            "SELECT match_id, is_home, possession_pct, ppda, "
            "shots_on_target, big_chances_created "
            "FROM match_stats_opta"
        ), conn)
    matches["kickoff_at"] = pd.to_datetime(matches["kickoff_at"], utc=True)
    return matches, stats


def _rolling_avg_for_team(
    team_id:           int,
    before_match_id:   int,
    matches:           pd.DataFrame,
    stats:             pd.DataFrame,
    n:                 int = ROLLING_N,
) -> Dict[str, Optional[float]]:
    """Calcula el rolling avg de stats Opta para team_id antes de before_match_id."""
    # Partidos del equipo (home o away) anteriores al partido objetivo
    team_matches = matches[
        ((matches["home_team_id"] == team_id) | (matches["away_team_id"] == team_id)) &
        (matches["match_id"] < before_match_id)
    ].sort_values("kickoff_at").tail(n)

    if team_matches.empty:
        return {col: np.nan for col in _OPTA_COLS}

    values: Dict[str, List[float]] = {col: [] for col in _OPTA_COLS}

    for _, m in team_matches.iterrows():
        mid     = m["match_id"]
        is_home_flag = (m["home_team_id"] == team_id)
        row = stats[
            (stats["match_id"] == mid) & (stats["is_home"] == is_home_flag)
        ]
        if row.empty:
            continue
        for col in _OPTA_COLS:
            val = row.iloc[0][col]
            if pd.notna(val):
                values[col].append(float(val))

    return {
        col: float(np.mean(v)) if v else np.nan
        for col, v in values.items()
    }


def compute_opta_features(engine) -> pd.DataFrame:
    """Devuelve DataFrame con match_id + 8 columnas familia G."""
    matches, stats = _load_data(engine)

    # Solo partidos que tienen stats Opta disponibles al menos parcialmente
    mids_with_stats = set(stats["match_id"].unique())
    # Partidos a enriquecer: aquellos donde el equipo tiene historial con stats
    # (pueden no tener stats propias si es el primer partido con Opta)
    rows: List[Dict] = []

    for _, match in matches.iterrows():
        mid     = int(match["match_id"])
        home_id = int(match["home_team_id"])
        away_id = int(match["away_team_id"])

        home_rolling = _rolling_avg_for_team(home_id, mid, matches, stats)
        away_rolling = _rolling_avg_for_team(away_id, mid, matches, stats)

        # Solo incluir si hay al menos una feature no-NaN para alguno de los dos equipos
        all_nan = all(
            np.isnan(v) for v in list(home_rolling.values()) + list(away_rolling.values())
            if v is not None
        )
        if all_nan:
            continue

        rows.append({
            "match_id":                mid,
            "home_possession_last5":   home_rolling["possession_pct"],
            "away_possession_last5":   away_rolling["possession_pct"],
            "home_ppda_last5":         home_rolling["ppda"],
            "away_ppda_last5":         away_rolling["ppda"],
            "home_shots_ot_last5":     home_rolling["shots_on_target"],
            "away_shots_ot_last5":     away_rolling["shots_on_target"],
            "home_bigchances_last5":   home_rolling["big_chances_created"],
            "away_bigchances_last5":   away_rolling["big_chances_created"],
        })

    logger.info("Opta features calculadas para %d partidos", len(rows))
    return pd.DataFrame(rows)


def write_to_db(df: pd.DataFrame, engine) -> None:
    """Escribe las features Opta en match_features via UPDATE."""
    if df.empty:
        logger.warning("DataFrame vacio, nada que escribir")
        return

    update_sql = text("""
        UPDATE match_features SET
            home_possession_last5  = :home_possession_last5,
            away_possession_last5  = :away_possession_last5,
            home_ppda_last5        = :home_ppda_last5,
            away_ppda_last5        = :away_ppda_last5,
            home_shots_ot_last5    = :home_shots_ot_last5,
            away_shots_ot_last5    = :away_shots_ot_last5,
            home_bigchances_last5  = :home_bigchances_last5,
            away_bigchances_last5  = :away_bigchances_last5
        WHERE match_id = :match_id
    """)

    with engine.begin() as conn:
        for row in df.to_dict(orient="records"):
            conn.execute(update_sql, row)

    logger.info("match_features actualizado para %d partidos", len(df))


if __name__ == "__main__":
    eng = create_engine(os.environ["DATABASE_URL"])
    df  = compute_opta_features(eng)
    write_to_db(df, eng)
    print(df.head())
