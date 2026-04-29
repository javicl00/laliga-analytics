"""Normalizador de partidos.

Fuente real verificada:
  Los partidos NO están en subscription.rounds[].gameweeks[].
  Los gameweeks solo contienen: id, week, name, shortname, date.

  Estrategia de extracción de partidos (en orden de preferencia):
  1. GET /subscriptions/{slug}/gameweek/{gameweek_id}/matches
     → Endpoint por jornada individual (a verificar)
  2. GET /subscriptions/{slug}/calendar
     → Endpoint de calendario completo (a verificar)
  3. Extracción desde payload de resultados por equipo

  Este módulo normaliza el formato esperado una vez confirmado el endpoint.
  Se implementa con duck typing sobre las claves conocidas.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


# Mapeo de status conocidos de la API a valores canónicos internos
STATUS_MAP = {
    "FinishedPeriod": "finished",
    "FullTime": "finished",
    "Finished": "finished",
    "PreMatch": "scheduled",
    "InProgress": "live",
    "Postponed": "postponed",
    "Cancelled": "cancelled",
    "Suspended": "suspended",
}


def normalize_match(raw_match: Dict, season_id: int, gameweek_id: int) -> Dict:
    """Normaliza un único partido al esquema interno de la tabla ``matches``.

    Acepta cualquiera de los formatos observados en la API:
    - Con claves snake_case: home_team, away_team, kickoff_at
    - Con claves camelCase: homeTeam, awayTeam, kickoffAt  (fallback)
    """
    home = raw_match.get("home_team") or raw_match.get("homeTeam") or {}
    away = raw_match.get("away_team") or raw_match.get("awayTeam") or {}

    # Score puede estar en score{}, result{} o directamente
    score = raw_match.get("score") or raw_match.get("result") or {}
    home_goals = (
        score.get("home") or score.get("homeGoals")
        or raw_match.get("home_goals") or raw_match.get("homeGoals")
    )
    away_goals = (
        score.get("away") or score.get("awayGoals")
        or raw_match.get("away_goals") or raw_match.get("awayGoals")
    )

    raw_status = raw_match.get("status") or raw_match.get("state") or ""
    canonical_status = STATUS_MAP.get(raw_status, raw_status.lower() if raw_status else None)

    kickoff = (
        raw_match.get("kickoff_at")
        or raw_match.get("kickoffAt")
        or raw_match.get("date")
        or raw_match.get("startTime")
    )

    venue = raw_match.get("venue") or raw_match.get("stadium") or {}

    return {
        "match_id": raw_match.get("id"),
        "season_id": season_id,
        "gameweek_id": gameweek_id,
        "kickoff_at": kickoff,
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_goals": int(home_goals) if home_goals is not None else None,
        "away_goals": int(away_goals) if away_goals is not None else None,
        "status": canonical_status,
        "venue_name": venue.get("name") if isinstance(venue, dict) else None,
    }


def normalize_matches_list(
    matches_payload: Any,
    season_id: int,
    gameweek_id: int,
) -> List[Dict]:
    """Normaliza una lista de partidos desde cualquier payload conocido."""
    # Intenta extraer la lista desde claves conocidas
    matches = (
        matches_payload.get("matches")
        or matches_payload.get("match_days")
        or matches_payload.get("events")
        or matches_payload.get("data")
        or []
    )
    # Si el payload es directamente una lista
    if isinstance(matches_payload, list):
        matches = matches_payload

    return [normalize_match(m, season_id, gameweek_id) for m in matches if isinstance(m, dict)]
