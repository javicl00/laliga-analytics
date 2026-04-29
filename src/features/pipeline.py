"""Pipeline completo: extrae matches de BD, computa features y las persiste."""
from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text

from src.features.builder import build_features
from src.storage.repository import PostgresRawRepository

logger = logging.getLogger(__name__)


def run(season_id: int | None = None, db_url: str | None = None) -> None:
    db_url = db_url or os.environ["DATABASE_URL"]
    engine = create_engine(db_url, pool_pre_ping=True)
    repo = PostgresRawRepository(db_url)

    query = """
        SELECT match_id, kickoff_at, gameweek_week,
               home_team_id, away_team_id,
               home_score, away_score, result
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
    features_df = build_features(df)
    logger.info("Computed features for %d matches", len(features_df))

    rows = features_df.to_dict(orient="records")
    repo.upsert_match_features(rows)
    logger.info("Feature pipeline complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
