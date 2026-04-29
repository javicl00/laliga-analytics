"""Normalizador de estadísticas de jugadores.

Fuente real: GET /subscriptions/{slug}/players/stats
Estructura verificada:
  total: int
  player_stats: [
    {
      id, name, nickname, slug,
      position: {id, name, ...},
      country: {id, name, ...},
      team: {id, name, slug, ...},
      shirt_number,
      opta_id,
      extra_info: {...},
      stats: [{name: X, stat: V}, ...]
    }
  ]

Diferencia con 2024: campo 'photos' presente en 2024, ausente en 2025.
Normalización agnóstica: ambas versiones funcionan igual.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


def _parse_stat_list(stats_list: List[Dict]) -> Dict[str, Any]:
    return {item["name"]: item["stat"] for item in stats_list if "name" in item and "stat" in item}


def normalize_player_stats(
    player_stats_payload: Any,
    season_id: int,
    gameweek_id: Optional[int],
    snapshot_ts: Optional[str] = None,
) -> List[Dict]:
    """Normaliza el payload de players/stats a filas para ``player_stats_snapshots``.

    Returns
    -------
    Lista de dicts listos para insertar en ``player_stats_snapshots``.
    """
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc).isoformat()

    players = player_stats_payload.get("player_stats", [])
    result = []
    for player in players:
        player_id = player.get("id")
        team = player.get("team") or {}
        team_id = team.get("id")
        stats_raw = player.get("stats", [])
        stats = _parse_stat_list(stats_raw) if isinstance(stats_raw, list) else stats_raw

        for stat_name, stat_value in stats.items():
            result.append({
                "season_id": season_id,
                "gameweek_id": gameweek_id,
                "player_id": player_id,
                "team_id": team_id,
                "snapshot_ts": snapshot_ts,
                "stat_name": stat_name,
                "stat_value": stat_value if isinstance(stat_value, (int, float)) else None,
            })
    return result


def normalize_players_master(player_stats_payload: Any) -> List[Dict]:
    """Extrae datos maestros de jugadores (para tabla ``players``)."""
    players = player_stats_payload.get("player_stats", [])
    result = []
    for p in players:
        position = p.get("position") or {}
        country = p.get("country") or {}
        team = p.get("team") or {}
        result.append({
            "player_id": p.get("id"),
            "opta_id": p.get("opta_id"),
            "slug": p.get("slug"),
            "name": p.get("name"),
            "nickname": p.get("nickname"),
            "shirt_number": p.get("shirt_number"),
            "position_id": position.get("id"),
            "position_name": position.get("name"),
            "country_id": country.get("id"),
            "country_name": country.get("name"),
            "team_id": team.get("id"),
        })
    return result
