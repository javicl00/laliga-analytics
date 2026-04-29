"""Normalizador de jornadas (gameweeks).

Fuente real: subscription.rounds[0].gameweeks[]
Estructura verificada:
  id, week, name, shortname, date

NOTA CRÍTICA: Los partidos NO están dentro de gameweeks.
Los partidos se obtienen por endpoint separado:
  GET /subscriptions/{slug}/gameweek/{gameweek_id}/matches
O bien via:
  GET /subscriptions/{slug}/calendar  (pendiente de verificar)
"""
from __future__ import annotations
from typing import Any, Dict, List


def normalize_gameweeks(subscription_payload: Any, season_id: int) -> List[Dict]:
    """Extrae y normaliza las jornadas desde subscription.rounds[].gameweeks[].

    Returns
    -------
    Lista de dicts listos para insertar en la tabla ``gameweeks``.
    """
    sub = subscription_payload.get("subscription", subscription_payload)
    rounds = sub.get("rounds", [])
    result = []
    seen = set()
    for r in rounds:
        for gw in r.get("gameweeks", []):
            gw_id = gw["id"]
            if gw_id in seen:
                continue
            seen.add(gw_id)
            result.append({
                "gameweek_id": gw_id,
                "season_id": season_id,
                "week": gw.get("week"),
                "name": gw.get("name"),
                "shortname": gw.get("shortname"),
                "date_start": gw.get("date"),  # ISO 8601
            })
    return result
