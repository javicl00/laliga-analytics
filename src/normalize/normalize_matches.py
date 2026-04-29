"""Normalizador de partidos.

Endpoint real verificado (HAR 2025-04-29):
  GET /public-service/api/v1/matches
      ?subscriptionSlug={slug}&week={week_number}&limit=100
      &orderField=date&orderType=asc

Estructura real del payload:
  total: int
  matches: [
    {
      id,
      name, slug,
      date,           → kickoff_at (ISO 8601 con tz)
      time,           → igual que date
      hashtag,
      competition: {id, name, slug, main, opta_id, lde_id},
      status,         → 'PreMatch' | 'FullTime' | 'FinishedPeriod' | ...
      home_score,     → GOLES LOCAL (incluido directamente, sin fan-out)
      away_score,     → GOLES VISITANTE
      home_team: {id, slug, name, shortname, color, ...},
      away_team: {id, slug, name, shortname, ...},
      match_winner:   {id, slug, ...} | null  (null si empate)
      home_formation, away_formation,
      gameweek: {id, week, name, shortname, date},
      venue: {id, name, latitude, longitude, capacity, ...},
      persons: [{role, person_id, name, role_id, ...}],  (arbitros)
      channels: [{id, name, ...}],
      season: {id, name, year, slug, ...},
      opta_id, lde_id,
      is_brand_day
    }
  ]

NOTA: home_score y away_score son None cuando status='PreMatch'.
NOTA: competition.main=true identifica la competicion principal (LaLiga).
        Usar para filtrar partidos de otras competiciones en el mismo payload.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


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


def normalize_match(raw: Dict, season_id: int) -> Dict:
    """Normaliza un partido al esquema interno de la tabla ``matches``.

    home_score y away_score se leen directamente del payload
    (no requieren fan-out a /matches/{id}).
    """
    home = raw.get("home_team") or {}
    away = raw.get("away_team") or {}
    gameweek = raw.get("gameweek") or {}
    venue = raw.get("venue") or {}
    competition = raw.get("competition") or {}
    winner = raw.get("match_winner") or {}

    raw_status = raw.get("status") or ""
    canonical_status = STATUS_MAP.get(raw_status, raw_status.lower() if raw_status else None)

    home_score = raw.get("home_score")
    away_score = raw.get("away_score")

    return {
        "match_id": raw.get("id"),
        "season_id": season_id,
        "gameweek_id": gameweek.get("id"),
        "gameweek_week": gameweek.get("week"),
        "kickoff_at": raw.get("date"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_score": int(home_score) if home_score is not None else None,
        "away_score": int(away_score) if away_score is not None else None,
        "result": (
            "home" if winner and winner.get("id") == home.get("id")
            else "away" if winner and winner.get("id") == away.get("id")
            else "draw" if canonical_status == "finished" and home_score is not None
            else None
        ),
        "status": canonical_status,
        "raw_status": raw_status,
        "home_formation": raw.get("home_formation"),
        "away_formation": raw.get("away_formation"),
        "competition_id": competition.get("id"),
        "competition_main": competition.get("main", False),
        "venue_id": venue.get("id") if isinstance(venue, dict) else None,
        "venue_name": venue.get("name") if isinstance(venue, dict) else None,
        "opta_id": raw.get("opta_id"),
        "lde_id": raw.get("lde_id"),
        "is_brand_day": raw.get("is_brand_day", False),
    }


def normalize_matches_page(
    matches_payload: Any,
    season_id: int,
    main_only: bool = True,
) -> List[Dict]:
    """Normaliza el payload de GET /matches?subscriptionSlug=...&week=...

    Parameters
    ----------
    matches_payload:
        JSON devuelto por el endpoint.
    season_id:
        ID interno de la temporada.
    main_only:
        Si True (por defecto), filtra solo partidos con competition.main=True.
        Elimina partidos de Copa, Champions, etc. que aparecen en el mismo slot.

    Returns
    -------
    Lista de dicts normalizados para insertar en ``matches``.
    """
    raw_list = matches_payload.get("matches", [])
    result = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        if main_only:
            comp = raw.get("competition") or {}
            if not comp.get("main", False):
                continue
        result.append(normalize_match(raw, season_id))
    return result
