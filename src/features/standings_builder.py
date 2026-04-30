"""Computa standings prepartido directamente desde el historial de resultados.

No requiere datos externos: deriva toda la clasificacion a partir de los
resultados en la tabla matches. La clasificacion se resetea por season_id.

Logica anti-leakage:
  - El snapshot para match_id M se calcula con TODOS los partidos
    finalizados con kickoff_at ESTRICTAMENTE anterior al kickoff de M.
  - El propio partido M nunca se incluye en su snapshot.

Criterios de clasificacion (LaLiga):
  1. Puntos DESC
  2. Diferencia de goles DESC
  3. Goles a favor DESC
  4. Orden alfabetico de team_id como desempate final (determinista)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estructuras internas
# ---------------------------------------------------------------------------

def _empty_stats(team_id: int) -> Dict:
    return {"team_id": team_id, "pts": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}


def _update_stats(stats: Dict, gf: int, ga: int) -> None:
    """Actualiza stats in-place para un equipo dado sus goles."""
    stats["gf"] += gf
    stats["ga"] += ga
    if gf > ga:
        stats["pts"] += 3
        stats["w"] += 1
    elif gf == ga:
        stats["pts"] += 1
        stats["d"] += 1
    else:
        stats["l"] += 1


def _get_position(team_stats: Dict[int, Dict], target_id: int) -> int:
    """Devuelve la posicion 1-based del equipo en la clasificacion actual."""
    rows = sorted(
        team_stats.values(),
        key=lambda r: (r["pts"], r["gf"] - r["ga"], r["gf"], -r["team_id"]),
        reverse=True,
    )
    for i, r in enumerate(rows, 1):
        if r["team_id"] == target_id:
            return i
    return len(rows)  # fallback (no deberia ocurrir si team_id esta en team_stats)


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def build_match_standings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve snapshot de clasificacion prepartido para cada partido.

    Parameters
    ----------
    matches_df : DataFrame con columnas:
        match_id, season_id, kickoff_at, home_team_id, away_team_id,
        home_score, away_score, result

    Returns
    -------
    DataFrame con match_id + columnas de standings:
        home_points_total, away_points_total,
        home_table_position, away_table_position, position_diff,
        home_gd_total, away_gd_total,
        home_pressure_index, away_pressure_index
    """
    df = matches_df.copy()
    df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True)
    df = df.sort_values(["season_id", "kickoff_at", "match_id"]).reset_index(drop=True)

    records: List[Dict] = []
    current_season = None
    team_stats: Dict[int, Dict] = {}
    total_gw = 38

    for _, row in df.iterrows():
        season_id = row.get("season_id")
        match_id  = int(row["match_id"])
        home_id   = int(row["home_team_id"])
        away_id   = int(row["away_team_id"])
        gw        = int(row.get("gameweek_week", 1) or 1)

        # Reset clasificacion al inicio de cada temporada y pre-pobla los 20 equipos
        if season_id != current_season:
            current_season = season_id
            team_stats = {}
            season_mask = df["season_id"] == season_id
            season_team_ids = set(
                pd.concat([
                    df.loc[season_mask, "home_team_id"],
                    df.loc[season_mask, "away_team_id"],
                ]).astype(int).unique()
            )
            for tid in season_team_ids:
                team_stats[tid] = _empty_stats(tid)
            logger.debug(
                "Season %s: reset standings, %d teams", season_id, len(team_stats)
            )

        # -- Snapshot ANTES del partido --
        h = team_stats.get(home_id, _empty_stats(home_id))
        a = team_stats.get(away_id, _empty_stats(away_id))
        h_gd  = h["gf"] - h["ga"]
        a_gd  = a["gf"] - a["ga"]
        h_pos = _get_position(team_stats, home_id)
        a_pos = _get_position(team_stats, away_id)
        remaining = max(0, total_gw - gw)

        def _pressure(pos: int) -> float:
            """Presion competitiva: alta en zona descenso (16-20) o titulo (1-4)."""
            if pos >= 16:
                return round(remaining / (remaining + 1.0) * (pos / 20.0), 4)
            elif pos <= 4:
                return round(remaining / (remaining + 1.0) * (1.0 - pos / 20.0), 4)
            return 0.1

        records.append({
            "match_id":            match_id,
            "home_points_total":   float(h["pts"]),
            "away_points_total":   float(a["pts"]),
            "home_table_position": h_pos,
            "away_table_position": a_pos,
            "position_diff":       float(h_pos - a_pos),
            "home_gd_total":       float(h_gd),
            "away_gd_total":       float(a_gd),
            "home_pressure_index": _pressure(h_pos),
            "away_pressure_index": _pressure(a_pos),
        })

        # -- Actualizar clasificacion DESPUES del snapshot (anti-leakage) --
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if pd.notna(row.get("result")) and pd.notna(home_score) and pd.notna(away_score):
            _update_stats(team_stats[home_id], int(home_score), int(away_score))
            _update_stats(team_stats[away_id], int(away_score), int(home_score))

    logger.info("Built standings snapshots for %d matches", len(records))
    return pd.DataFrame(records)
