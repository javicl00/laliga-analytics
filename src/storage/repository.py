"""Repository pattern para persistencia raw y normalizada."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class PostgresRawRepository:
    """Persiste payloads JSON en raw_snapshots y filas normalizadas en sus tablas."""

    def __init__(self, db_url: str | None = None) -> None:
        self._url = db_url or os.environ["DATABASE_URL"]
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._url, pool_pre_ping=True)
        return self._engine

    # ------------------------------------------------------------------ raw

    def save(
        self,
        resource: str,
        payload: Any,
        competition_slug: str,
        season_label: str,
    ) -> None:
        sql = text("""
            INSERT INTO raw_snapshots (resource, competition_slug, season_label, payload)
            VALUES (:resource, :competition_slug, :season_label, :payload)
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, {
                "resource": resource,
                "competition_slug": competition_slug,
                "season_label": season_label,
                "payload": json.dumps(payload, ensure_ascii=False),
            })
        logger.debug("Saved raw_snapshot resource=%s", resource)

    # ------------------------------------------------------------------ upserts

    def upsert_teams(self, teams: List[Dict]) -> None:
        sql = text("""
            INSERT INTO teams (team_id, slug, name, shortname, color, opta_id, lde_id)
            VALUES (:team_id, :slug, :name, :shortname, :color, :opta_id, :lde_id)
            ON CONFLICT (team_id) DO UPDATE SET
                slug=EXCLUDED.slug, name=EXCLUDED.name,
                shortname=EXCLUDED.shortname, color=EXCLUDED.color
        """)
        with self.engine.begin() as conn:
            for t in teams:
                conn.execute(sql, t)
        logger.info("Upserted %d teams", len(teams))

    def upsert_season(self, season: Dict) -> None:
        sql = text("""
            INSERT INTO seasons (season_id, name, year, slug)
            VALUES (:season_id, :name, :year, :slug)
            ON CONFLICT (season_id) DO NOTHING
        """)
        with self.engine.begin() as conn:
            conn.execute(sql, season)

    def upsert_gameweeks(self, gameweeks: List[Dict]) -> None:
        sql = text("""
            INSERT INTO gameweeks (gameweek_id, season_id, week, name, date)
            VALUES (:gameweek_id, :season_id, :week, :name, :date)
            ON CONFLICT (gameweek_id) DO NOTHING
        """)
        with self.engine.begin() as conn:
            for gw in gameweeks:
                conn.execute(sql, gw)

    def upsert_matches(self, matches: List[Dict]) -> None:
        sql = text("""
            INSERT INTO matches (
                match_id, season_id, gameweek_id, gameweek_week, kickoff_at,
                home_team_id, away_team_id, home_score, away_score, result,
                status, raw_status, home_formation, away_formation,
                competition_id, competition_main, venue_id, venue_name,
                opta_id, lde_id, is_brand_day
            ) VALUES (
                :match_id, :season_id, :gameweek_id, :gameweek_week, :kickoff_at,
                :home_team_id, :away_team_id, :home_score, :away_score, :result,
                :status, :raw_status, :home_formation, :away_formation,
                :competition_id, :competition_main, :venue_id, :venue_name,
                :opta_id, :lde_id, :is_brand_day
            )
            ON CONFLICT (match_id) DO UPDATE SET
                home_score=EXCLUDED.home_score,
                away_score=EXCLUDED.away_score,
                result=EXCLUDED.result,
                status=EXCLUDED.status,
                raw_status=EXCLUDED.raw_status,
                home_formation=EXCLUDED.home_formation,
                away_formation=EXCLUDED.away_formation
        """)
        with self.engine.begin() as conn:
            for m in matches:
                conn.execute(sql, m)
        logger.info("Upserted %d matches", len(matches))

    def upsert_standing(self, rows: List[Dict]) -> None:
        sql = text("""
            INSERT INTO standings (
                season_id, team_id, position, points, played, won, drawn, lost,
                goals_for, goals_against, goal_difference, qualify_name
            ) VALUES (
                :season_id, :team_id, :position, :points, :played, :won, :drawn, :lost,
                :goals_for, :goals_against, :goal_difference, :qualify_name
            )
        """)
        with self.engine.begin() as conn:
            for r in rows:
                conn.execute(sql, r)
        logger.info("Inserted %d standing rows", len(rows))

    def upsert_match_features(self, rows: List[Dict]) -> None:
        """Upsert de features ricas v2 (schema match_features v2).

        Columnas esperadas en cada dict: match_id + las 19 de FEATURE_COLUMNS
        definidas en src/features/build_features.py.
        """
        sql = text("""
            INSERT INTO match_features (
                match_id,
                home_elo, away_elo, elo_diff,
                home_points_total, away_points_total,
                home_table_position, away_table_position, position_diff,
                home_gd_total, away_gd_total,
                home_goals_for_last5, home_goals_against_last5,
                away_goals_for_last5, away_goals_against_last5,
                gameweek, home_rest_days, away_rest_days,
                home_pressure_index, away_pressure_index
            ) VALUES (
                :match_id,
                :home_elo, :away_elo, :elo_diff,
                :home_points_total, :away_points_total,
                :home_table_position, :away_table_position, :position_diff,
                :home_gd_total, :away_gd_total,
                :home_goals_for_last5, :home_goals_against_last5,
                :away_goals_for_last5, :away_goals_against_last5,
                :gameweek, :home_rest_days, :away_rest_days,
                :home_pressure_index, :away_pressure_index
            )
            ON CONFLICT (match_id) DO UPDATE SET
                home_elo=EXCLUDED.home_elo,
                away_elo=EXCLUDED.away_elo,
                elo_diff=EXCLUDED.elo_diff,
                home_points_total=EXCLUDED.home_points_total,
                away_points_total=EXCLUDED.away_points_total,
                home_table_position=EXCLUDED.home_table_position,
                away_table_position=EXCLUDED.away_table_position,
                position_diff=EXCLUDED.position_diff,
                home_gd_total=EXCLUDED.home_gd_total,
                away_gd_total=EXCLUDED.away_gd_total,
                home_goals_for_last5=EXCLUDED.home_goals_for_last5,
                home_goals_against_last5=EXCLUDED.home_goals_against_last5,
                away_goals_for_last5=EXCLUDED.away_goals_for_last5,
                away_goals_against_last5=EXCLUDED.away_goals_against_last5,
                gameweek=EXCLUDED.gameweek,
                home_rest_days=EXCLUDED.home_rest_days,
                away_rest_days=EXCLUDED.away_rest_days,
                home_pressure_index=EXCLUDED.home_pressure_index,
                away_pressure_index=EXCLUDED.away_pressure_index,
                computed_at=now()
        """)
        with self.engine.begin() as conn:
            for r in rows:
                conn.execute(sql, r)
        logger.info("Upserted %d match_features rows", len(rows))

    def save_predictions(self, rows: List[Dict]) -> None:
        sql = text("""
            INSERT INTO predictions (match_id, model_name, model_version, prob_home, prob_draw, prob_away)
            VALUES (:match_id, :model_name, :model_version, :prob_home, :prob_draw, :prob_away)
        """)
        with self.engine.begin() as conn:
            for r in rows:
                conn.execute(sql, r)
        logger.info("Saved %d predictions", len(rows))

    def fetch_all_matches(self, season_id: int | None = None) -> List[Dict]:
        sql = "SELECT * FROM matches WHERE competition_main = TRUE"
        params: Dict = {}
        if season_id:
            sql += " AND season_id = :season_id"
            params["season_id"] = season_id
        sql += " ORDER BY kickoff_at"
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]
