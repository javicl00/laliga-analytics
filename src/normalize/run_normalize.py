"""Normalizador: raw_snapshots -> tablas relacionales.

Lee los payloads JSON de raw_snapshots y puebla:
  - teams
  - seasons
  - gameweeks   (desde subscription Y desde los propios partidos)
  - matches
  - standings

Uso:
    python -m src.normalize.run_normalize
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers

def _result(home: Optional[int], away: Optional[int]) -> Optional[str]:
    if home is None or away is None:
        return None
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "draw"


def _int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ parsers

def parse_subscription(payload: Any) -> Dict:
    sub = payload.get("subscription") or payload

    season_slug = sub.get("slug", "")
    year = None
    for part in season_slug.split("-"):
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break

    season = {
        "season_id": sub.get("id"),
        "name":      sub.get("season_name") or sub.get("season") or "",
        "year":      year or 2025,
        "slug":      season_slug,
    }

    teams_raw = sub.get("teams") or []
    teams = []
    for t in teams_raw:
        teams.append({
            "team_id":   _int(t.get("id")),
            "slug":      t.get("slug", ""),
            "name":      t.get("name") or t.get("nickname") or "",
            "shortname": t.get("shortname") or t.get("nickname"),
            "color":     t.get("color"),
            "opta_id":   str(t.get("opta_id")) if t.get("opta_id") else None,
            "lde_id":    _int(t.get("lde_id")),
        })

    # Gameweeks desde subscription (puede ser incompleto)
    gameweeks_raw = sub.get("gameweeks") or sub.get("rounds") or []
    gameweeks = []
    for gw in gameweeks_raw:
        gw_date = gw["start_date"][:10] if gw.get("start_date") else None
        gameweeks.append({
            "gameweek_id": _int(gw.get("id")),
            "season_id":   season["season_id"],
            "week":        _int(gw.get("week") or gw.get("round")),
            "name":        gw.get("name"),
            "date":        gw_date,
        })

    return {"season": season, "teams": teams, "gameweeks": gameweeks}


def extract_gameweeks_from_matches(snapshots, season_id: int) -> List[Dict]:
    """Extrae gameweeks unicos embebidos en los payloads de partidos.

    Cada partido tiene un campo 'gameweek': {id, week, name, ...}.
    Esta funcion los consolida en una lista deduplicada para
    garantizar que todos los gameweek_id existen antes de insertar matches.
    """
    seen = set()
    gameweeks = []
    for row in snapshots:
        if not row["resource"].startswith("matches_week_"):
            continue
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        for m in payload.get("matches", []):
            gw = m.get("gameweek") or {}
            gw_id = _int(gw.get("id"))
            if not gw_id or gw_id in seen:
                continue
            seen.add(gw_id)
            gw_date = gw.get("start_date") or gw.get("date")
            if gw_date:
                gw_date = gw_date[:10]
            gameweeks.append({
                "gameweek_id": gw_id,
                "season_id":   season_id,
                "week":        _int(gw.get("week") or gw.get("round")),
                "name":        gw.get("name"),
                "date":        gw_date,
            })
    logger.info("Extracted %d unique gameweeks from match payloads", len(gameweeks))
    return gameweeks


def parse_standing(payload: Any, season_id: int) -> List[Dict]:
    rows = []
    for raw in payload.get("standings", []):
        team = raw.get("team") or {}
        qualify = raw.get("qualify") or {}
        try:
            gd = int(str(raw.get("goal_difference", "0")).replace("+", ""))
        except (ValueError, TypeError):
            gd = 0
        rows.append({
            "season_id":       season_id,
            "team_id":         _int(team.get("id")),
            "position":        _int(raw.get("position")),
            "points":          _int(raw.get("points")),
            "played":          _int(raw.get("played")),
            "won":             _int(raw.get("won")),
            "drawn":           _int(raw.get("drawn")),
            "lost":            _int(raw.get("lost")),
            "goals_for":       _int(raw.get("goals_for")),
            "goals_against":   _int(raw.get("goals_against")),
            "goal_difference": gd,
            "qualify_name":    qualify.get("name"),
        })
    return rows


def parse_matches_week(payload: Any, season_id: int, week: int) -> List[Dict]:
    rows = []
    for m in payload.get("matches", []):
        competition = m.get("competition") or {}
        venue = m.get("venue") or {}
        home_score = _int(m.get("home_score"))
        away_score = _int(m.get("away_score"))
        gw = m.get("gameweek") or {}
        rows.append({
            "match_id":        _int(m.get("id")),
            "season_id":       season_id,
            "gameweek_id":     _int(gw.get("id")),
            "gameweek_week":   _int(gw.get("week")) or week,
            "kickoff_at":      m.get("kickoff_at") or m.get("date"),
            "home_team_id":    _int((m.get("home_team") or {}).get("id")),
            "away_team_id":    _int((m.get("away_team") or {}).get("id")),
            "home_score":      home_score,
            "away_score":      away_score,
            "result":          _result(home_score, away_score),
            "status":          m.get("status"),
            "raw_status":      m.get("status"),
            "home_formation":  m.get("home_formation"),
            "away_formation":  m.get("away_formation"),
            "competition_id":  _int(competition.get("id")),
            "competition_main": bool(competition.get("main", True)),
            "venue_id":        _int(venue.get("id")),
            "venue_name":      venue.get("name"),
            "opta_id":         str(m.get("opta_id")) if m.get("opta_id") else None,
            "lde_id":          _int(m.get("lde_id")),
            "is_brand_day":    bool(m.get("is_brand_day", False)),
        })
    return rows


# ------------------------------------------------------------------ upserts

def upsert_season(conn, season: Dict) -> None:
    if not season.get("season_id"):
        return
    conn.execute(text("""
        INSERT INTO seasons (season_id, name, year, slug)
        VALUES (:season_id, :name, :year, :slug)
        ON CONFLICT (season_id) DO NOTHING
    """), season)


def upsert_teams(conn, teams: List[Dict]) -> None:
    for t in teams:
        if not t.get("team_id"):
            continue
        conn.execute(text("""
            INSERT INTO teams (team_id, slug, name, shortname, color, opta_id, lde_id)
            VALUES (:team_id, :slug, :name, :shortname, :color, :opta_id, :lde_id)
            ON CONFLICT (team_id) DO UPDATE SET
                name=EXCLUDED.name, shortname=EXCLUDED.shortname, color=EXCLUDED.color
        """), t)


def upsert_gameweeks(conn, gameweeks: List[Dict]) -> None:
    for gw in gameweeks:
        if not gw.get("gameweek_id") or not gw.get("week"):
            continue
        conn.execute(text("""
            INSERT INTO gameweeks (gameweek_id, season_id, week, name, date)
            VALUES (:gameweek_id, :season_id, :week, :name, :date)
            ON CONFLICT (gameweek_id) DO NOTHING
        """), gw)


def upsert_matches(conn, matches: List[Dict]) -> None:
    for m in matches:
        if not m.get("match_id") or not m.get("home_team_id") or not m.get("away_team_id"):
            continue
        conn.execute(text("""
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
                gameweek_week=EXCLUDED.gameweek_week
        """), m)


def insert_standings(conn, rows: List[Dict]) -> None:
    for r in rows:
        if not r.get("team_id"):
            continue
        conn.execute(text("""
            INSERT INTO standings (
                season_id, team_id, position, points, played, won, drawn, lost,
                goals_for, goals_against, goal_difference, qualify_name
            ) VALUES (
                :season_id, :team_id, :position, :points, :played, :won, :drawn, :lost,
                :goals_for, :goals_against, :goal_difference, :qualify_name
            )
        """), r)


# ------------------------------------------------------------------ runner

def run(db_url: str | None = None) -> None:
    db_url = db_url or os.environ["DATABASE_URL"]
    engine = create_engine(db_url, pool_pre_ping=True)

    with engine.connect() as conn:
        snapshots = conn.execute(text(
            "SELECT resource, payload FROM raw_snapshots ORDER BY resource"
        )).mappings().all()
    snapshots = list(snapshots)

    logger.info("Found %d raw snapshots to normalize", len(snapshots))

    season_id: int | None = None

    # Paso 1: subscription -> season + teams + gameweeks (de subscription)
    for row in snapshots:
        if row["resource"] == "subscription":
            payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            sub_data = parse_subscription(payload)
            season_id = sub_data["season"]["season_id"]
            with engine.begin() as conn:
                upsert_season(conn, sub_data["season"])
                upsert_teams(conn, sub_data["teams"])
                upsert_gameweeks(conn, sub_data["gameweeks"])
            logger.info("Upserted season=%s teams=%d gameweeks_from_sub=%d",
                        season_id, len(sub_data["teams"]), len(sub_data["gameweeks"]))
            break

    if not season_id:
        logger.error("No subscription snapshot found — cannot normalize")
        return

    # Paso 1b: extraer gameweeks embebidos en los partidos (fuente de verdad real)
    match_gameweeks = extract_gameweeks_from_matches(snapshots, season_id)
    with engine.begin() as conn:
        upsert_gameweeks(conn, match_gameweeks)

    # Paso 2: matches_week_* -> matches
    total_matches = 0
    for row in snapshots:
        resource = row["resource"]
        if not resource.startswith("matches_week_"):
            continue
        week = int(resource.split("_")[-1])
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        matches = parse_matches_week(payload, season_id, week)
        with engine.begin() as conn:
            upsert_matches(conn, matches)
        total_matches += len(matches)
        logger.info("Week %2d: %d matches", week, len(matches))

    logger.info("Total matches upserted: %d", total_matches)

    # Paso 3: standing -> standings
    for row in snapshots:
        if row["resource"] == "standing":
            payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            standing_rows = parse_standing(payload, season_id)
            with engine.begin() as conn:
                insert_standings(conn, standing_rows)
            logger.info("Inserted %d standing rows", len(standing_rows))
            break

    logger.info("Normalization complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
