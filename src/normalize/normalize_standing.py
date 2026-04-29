"""Normalizador de clasificación.

Fuente real: GET /subscriptions/{slug}/standing
Estructura verificada:
  total: int
  standings: [
    {
      played, points, won, drawn, lost,
      goals_for, goals_against, goal_difference (str),
      position, previous_position, difference_position,
      team: {id, slug, name, shortname, ...}
    }
  ]

NOTA: goal_difference viene como string (ej: "+5", "-3"), se convierte a int.
NOTA: El endpoint devuelve la clasificación ACTUAL (no por jornada).
        Para snapshots históricos hay que llamar con cada gameweek_id.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def _parse_gd(value: Any) -> Optional[int]:
    """Convierte goal_difference a int (puede venir como str '+5' o '-3')."""
    if value is None:
        return None
    try:
        return int(str(value).replace("+", ""))
    except (ValueError, TypeError):
        return None


def normalize_standing(
    standing_payload: Any,
    season_id: int,
    gameweek_id: int,
    snapshot_ts: str,
) -> List[Dict]:
    """Normaliza el payload de standing a filas para ``standings_snapshots``.

    Parameters
    ----------
    standing_payload:
        JSON devuelto por GET /subscriptions/{slug}/standing.
    season_id:
        ID interno de la temporada.
    gameweek_id:
        ID de la jornada a la que corresponde este snapshot.
    snapshot_ts:
        Timestamp ISO 8601 del momento de extracción.

    Returns
    -------
    Lista de dicts listos para insertar en ``standings_snapshots``.
    """
    rows = standing_payload.get("standings", [])
    result = []
    for row in rows:
        team = row.get("team") or {}
        result.append({
            "season_id": season_id,
            "gameweek_id": gameweek_id,
            "team_id": team.get("id"),
            "snapshot_ts": snapshot_ts,
            "points": row.get("points"),
            "position": row.get("position"),
            "won": row.get("won"),
            "drawn": row.get("drawn"),
            "lost": row.get("lost"),
            "goals_for": row.get("goals_for"),
            "goals_against": row.get("goals_against"),
            "goal_difference": _parse_gd(row.get("goal_difference")),
            "played": row.get("played"),
            "previous_position": row.get("previous_position"),
        })
    return result
