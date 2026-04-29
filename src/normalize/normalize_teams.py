"""Normalizador de equipos maestros.

Fuente real: subscription.teams[]
Estructura verificada:
  id, slug, name, nickname, boundname, shortname,
  color, color_secondary, foundation, web,
  shield{url}, competitions[], opta_id, lde_id
"""
from __future__ import annotations
from typing import Any, Dict, List


def normalize_teams(subscription_payload: Any) -> List[Dict]:
    """Extrae y normaliza la lista de equipos desde el payload de subscription.

    Returns
    -------
    Lista de dicts listos para insertar en la tabla ``teams``.
    """
    sub = subscription_payload.get("subscription", subscription_payload)
    raw_teams = sub.get("teams", [])
    result = []
    for t in raw_teams:
        result.append({
            "team_id": t["id"],
            "slug": t.get("slug"),
            "name": t.get("name"),
            "shortname": t.get("shortname"),
            "nickname": t.get("nickname"),
            "opta_id": t.get("opta_id"),
            "lde_id": t.get("lde_id"),
            "color": t.get("color"),
            "color_secondary": t.get("color_secondary"),
            "foundation": t.get("foundation"),
            "web": t.get("web"),
            "shield_url": (t.get("shield") or {}).get("url"),
        })
    return result
