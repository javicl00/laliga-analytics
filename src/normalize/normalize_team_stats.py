"""Normalizador de estadísticas de equipo.

Fuente real: GET /subscriptions/{slug}/teams/stats
Estructura verificada:
  total: int
  team_stats: [
    {
      id, name, short_name, nick_name, slug,
      shield: {...},
      opta_id, lde_id,
      stats: [
        {name: "goals", stat: 36},
        {name: "possession_percentage", stat: 48.7},
        ...
      ]
    }
  ]

Stats clave verificadas (lista completa ~80 stats por equipo):
  goals, goals_conceded, points, position, games_played,
  won, drawn, lost, possession_percentage, ppda,
  total_shots, shots_on_target_inc_goals, passing_accuracy,
  clean_sheets, aerial_duels_won, duels_won, ...
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


# Subset de stats relevantes para features ML (anti-leakage: season-to-date)
STATS_FOR_ML = {
    "goals", "goals_conceded", "points", "position", "games_played",
    "won", "drawn", "lost", "possession_percentage", "ppda",
    "total_shots", "shots_on_target_inc_goals", "passing_accuracy",
    "clean_sheets", "aerial_duels_won", "aerial_duels_lost",
    "duels_won", "duels_lost", "goal_assists", "goal_conversion",
    "shooting_accuracy", "corners_won", "yellow_cards", "total_red_cards",
    "home_goals", "away_goals", "goals_openplay",
    "successful_dribbles", "unsuccessful_dribbles",
    "total_fouls_conceded", "total_fouls_won",
    "hit_woodwork", "blocked_shots", "interceptions",
    "key_passes_attempt_assists", "total_clearances",
    "successful_long_passes", "unsuccessful_long_passes",
    "penalties_taken", "penalty_goals", "penalties_conceded",
    "points_dropped_from_winning_positions",
    "points_gained_from_losing_positions",
}


def _parse_stat_list(stats_list: List[Dict]) -> Dict[str, Any]:
    """Convierte [{name: X, stat: V}] a {X: V}."""
    return {item["name"]: item["stat"] for item in stats_list if "name" in item and "stat" in item}


def normalize_team_stats(
    team_stats_payload: Any,
    season_id: int,
    gameweek_id: Optional[int],
    snapshot_ts: Optional[str] = None,
    ml_only: bool = False,
) -> List[Dict]:
    """Normaliza el payload de teams/stats a filas para ``team_stats_snapshots``.

    Parameters
    ----------
    team_stats_payload:
        JSON devuelto por GET /subscriptions/{slug}/teams/stats.
    season_id:
        ID interno de la temporada.
    gameweek_id:
        ID de la jornada asociada al snapshot (None = temporada completa).
    snapshot_ts:
        Timestamp ISO 8601. Si es None, se usa now().
    ml_only:
        Si True, solo persiste las stats de ``STATS_FOR_ML``.

    Returns
    -------
    Lista de dicts listos para insertar en ``team_stats_snapshots``.
    """
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc).isoformat()

    teams = team_stats_payload.get("team_stats", [])
    result = []
    for team in teams:
        team_id = team.get("id")
        stats_raw = team.get("stats", [])
        stats = _parse_stat_list(stats_raw) if isinstance(stats_raw, list) else stats_raw

        for stat_name, stat_value in stats.items():
            if ml_only and stat_name not in STATS_FOR_ML:
                continue
            result.append({
                "season_id": season_id,
                "gameweek_id": gameweek_id,
                "team_id": team_id,
                "snapshot_ts": snapshot_ts,
                "stat_name": stat_name,
                "stat_value": stat_value if isinstance(stat_value, (int, float)) else None,
                "scope": "season_to_date",
            })
    return result


def team_stats_to_wide(
    team_stats_payload: Any,
    stats_subset: Optional[List[str]] = None,
) -> List[Dict]:
    """Convierte team_stats a formato ancho: una fila por equipo con columnas de stats.

    Útil para feature building directo sin pasar por la BD.
    """
    teams = team_stats_payload.get("team_stats", [])
    subset = set(stats_subset) if stats_subset else None
    result = []
    for team in teams:
        row: Dict[str, Any] = {
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "team_slug": team.get("slug"),
            "opta_id": team.get("opta_id"),
        }
        stats_raw = team.get("stats", [])
        stats = _parse_stat_list(stats_raw) if isinstance(stats_raw, list) else stats_raw
        for k, v in stats.items():
            if subset is None or k in subset:
                row[k] = v
        result.append(row)
    return result
