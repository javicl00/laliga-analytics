"""Normalizador de clasificacion.

Endpoint: GET /subscriptions/{slug}/standing
Clave raiz: 'standings'
Campos clave: position, points, played, won, drawn, lost,
               goals_for, goals_against, goal_difference,
               team.id, qualify.name
"""
from __future__ import annotations
from typing import Any, Dict, List


def normalize_standing(payload: Any, season_id: int) -> List[Dict]:
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
            "team_id":         team.get("id"),
            "position":        raw.get("position"),
            "points":          raw.get("points"),
            "played":          raw.get("played"),
            "won":             raw.get("won"),
            "drawn":           raw.get("drawn"),
            "lost":            raw.get("lost"),
            "goals_for":       raw.get("goals_for"),
            "goals_against":   raw.get("goals_against"),
            "goal_difference": gd,
            "qualify_name":    qualify.get("name"),
        })
    return rows
