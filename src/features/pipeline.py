"""Pipeline completo: extrae matches de BD, computa features y las persiste.

Aplica migraciones SQL pendientes al arrancar (no requiere psql).

Usa FeatureBuilder (build_features.py) que calcula 22 features ricas:
  D - ELO dinamico (home_elo, away_elo, elo_diff)
  A - Estado competitivo desde standings snapshot (home/away_points_total,
      home/away_table_position, position_diff, home/away_gd_total)
  B - Forma reciente ultimos 5, todos los campos
      (home/away_goals_for/against_last5)
  E - Contexto: gameweek, home/away_rest_days, home/away_pressure_index
  F - Head-to-Head: h2h_home_wins, h2h_draws, h2h_away_wins

Anti-leakage: todas las features usan datos PREVIOS al kickoff del partido.
Procesa todos los partidos (con y sin resultado) para soportar inferencia.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import create_engine, text

from src.features.build_features import FeatureBuilder, FEATURE_COLUMNS
from src.storage.migrations import apply_migrations
from src.storage.repository import PostgresRawRepository

logger = logging.getLogger(__name__)


def _sanitize(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte float NaN/Inf a None para compatibilidad con psycopg2."""
    result = {}
    for k, v in row.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            result[k] = None
        else:
            result[k] = v
    return result


def run(season_id: int | None = None, db_url: str | None = None) -> None:
    db_url = db_url or os.environ["DATABASE_URL"]
    engine = create_engine(db_url, pool_pre_ping=True)
    repo = PostgresRawRepository(db_url)

    # Aplicar migraciones SQL pendientes antes de cualquier operacion
    apply_migrations(engine)

    query = """
        SELECT match_id, season_id, kickoff_at, gameweek_week,
               home_team_id, away_team_id,
               home_score, away_score, result, status
        FROM matches
        WHERE competition_main = TRUE
    """
    params: dict = {}
    if season_id:
        query += " AND season_id = :season_id"
        params["season_id"] = season_id
    query += " ORDER BY kickoff_at"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    logger.info("Loaded %d matches for feature engineering", len(df))
    features_df = FeatureBuilder(df).build()
    logger.info("Computed features for %d matches", len(features_df))

    cols = ["match_id"] + FEATURE_COLUMNS
    rows: List[Dict] = [
        _sanitize(r)
        for r in features_df[cols].to_dict(orient="records")
    ]
    repo.upsert_match_features(rows)
    logger.info("Feature pipeline complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
