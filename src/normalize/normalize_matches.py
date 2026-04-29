"""Normalizador de partidos.

Endpoint real verificado:
  GET /matches?subscriptionId={slug}&gameweekId={gameweek_id}

Estructura verificada (2025-04-29):
  total: int
  matches: [
    {
      id,              → match_id
      name, slug,
      date,            → kickoff_at (ISO 8601 con tz)
      time,            → igual que date
      hashtag,
      competition: {id, name, slug, main, opta_id, lde_id},
      status,          → 'PreMatch', 'FinishedPeriod', 'FullTime', etc.
      home_team: {id, slug, name, shortname, ...},
      away_team: {id, slug, name, shortname, ...},
      gameweek: {id, week, name, shortname, date},
      venue: {id, name, ...},
      season: {id, year, name, ...},
      is_brand_day,
      opta_id,
      lde_id
    }
  ]

NOTA: El endpoint devuelve partidos de TODAS las competiciones si no se filtra.
       Filtrar siempre por competition_id de LaLiga para evitar ruido.
       Competition LaLiga EA Sports = id a confirmar por suscripción.

NOTA: score/goles NO aparece en el payload de /matches (solo metadata).
       Los goles se obtienen desde un endpoint de detalle por partido:
       GET /matches/{match_id}  → pendiente de verificar.
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

    IMPORTANTE: home_goals y away_goals son None hasta que se obtenga
    el detalle del partido desde GET /matches/{id}.
    """
    home = raw.get("home_team") or {}
    away = raw.get("away_team") or {}
    gameweek = raw.get("gameweek") or {}
    venue = raw.get("venue") or {}
    competition = raw.get("competition") or {}

    raw_status = raw.get("status") or ""
    canonical_status = STATUS_MAP.get(raw_status, raw_status.lower() if raw_status else None)

    return {
        "match_id": raw.get("id"),
        "season_id": season_id,
        "gameweek_id": gameweek.get("id"),
        "gameweek_week": gameweek.get("week"),
        "kickoff_at": raw.get("date"),          # ISO 8601 con tz
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_goals": None,                     # Requiere GET /matches/{id}
        "away_goals": None,
        "status": canonical_status,
        "raw_status": raw_status,
        "competition_id": competition.get("id"),
        "competition_slug": competition.get("slug"),
        "venue_id": venue.get("id") if isinstance(venue, dict) else None,
        "venue_name": venue.get("name") if isinstance(venue, dict) else None,
        "opta_id": raw.get("opta_id"),
        "lde_id": raw.get("lde_id"),
        "is_brand_day": raw.get("is_brand_day", False),
    }


def normalize_matches_page(
    matches_payload: Any,
    season_id: int,
    laliga_competition_id: Optional[int] = None,
) -> List[Dict]:
    """Normaliza un payload de GET /matches?subscriptionId=...&gameweekId=...

    Parameters
    ----------
    matches_payload:
        JSON devuelto por el endpoint.
    season_id:
        ID interno de la temporada.
    laliga_competition_id:
        Si se especifica, filtra solo partidos de esa competición.
        Usar para excluir Copa, Champions, etc. del mismo payload.

    Returns
    -------
    Lista de dicts normalizados para insertar en ``matches``.
    """
    raw_list = matches_payload.get("matches", [])
    result = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        # Filtro por competición si se especifica
        if laliga_competition_id is not None:
            comp_id = (raw.get("competition") or {}).get("id")
            if comp_id != laliga_competition_id:
                continue
        result.append(normalize_match(raw, season_id))
    return result
