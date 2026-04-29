"""Feature engineering para el modelo de prediccion de partidos.

Todas las ventanas se calculan ESTRICTAMENTE antes de kickoff_at
de cada partido para evitar data leakage.

Features calculadas por partido:
  - home_form_pts / away_form_pts      : puntos en ultimas WINDOW jornadas
  - home_gf_avg  / home_gc_avg         : goles marcados/recibidos (ventana)
  - away_gf_avg  / away_gc_avg
  - home_position / away_position      : posicion en tabla antes del partido
  - h2h_home_wins / h2h_draws / h2h_away_wins : H2H ultimas H2H_MATCHES
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

WINDOW = 5       # ultimos N partidos para forma y goles medios
H2H_MATCHES = 10 # ultimos N enfrentamientos H2H


def _points(home_score, away_score, team_side: str) -> int:
    if home_score is None or away_score is None:
        return 0
    if home_score == away_score:
        return 1
    if team_side == "home":
        return 3 if home_score > away_score else 0
    return 3 if away_score > home_score else 0


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Genera features para todos los partidos terminados.

    Parameters
    ----------
    matches_df : DataFrame con columnas:
        match_id, kickoff_at, home_team_id, away_team_id,
        home_score, away_score, result, gameweek_week

    Returns
    -------
    DataFrame con match_id + todas las features.
    """
    df = matches_df.copy()
    df = df.sort_values("kickoff_at").reset_index(drop=True)
    df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True)

    finished = df[df["result"].notna()].copy()

    records: List[Dict] = []
    for _, row in df.iterrows():
        if pd.isna(row["result"]):
            continue
        feat = _compute_features_for_match(row, finished)
        records.append(feat)

    return pd.DataFrame(records)


def _compute_features_for_match(row: pd.Series, finished: pd.DataFrame) -> Dict:
    mid      = int(row["match_id"])
    home_id  = int(row["home_team_id"])
    away_id  = int(row["away_team_id"])
    kickoff  = row["kickoff_at"]

    # Partidos previos para cada equipo (antes de este partido)
    home_prev = finished[
        ((finished["home_team_id"] == home_id) | (finished["away_team_id"] == home_id))
        & (finished["kickoff_at"] < kickoff)
    ].tail(WINDOW)

    away_prev = finished[
        ((finished["home_team_id"] == away_id) | (finished["away_team_id"] == away_id))
        & (finished["kickoff_at"] < kickoff)
    ].tail(WINDOW)

    # H2H
    h2h = finished[
        (
            ((finished["home_team_id"] == home_id) & (finished["away_team_id"] == away_id))
            | ((finished["home_team_id"] == away_id) & (finished["away_team_id"] == home_id))
        )
        & (finished["kickoff_at"] < kickoff)
    ].tail(H2H_MATCHES)

    return {
        "match_id":       mid,
        "home_form_pts":  _form_pts(home_prev, home_id),
        "away_form_pts":  _form_pts(away_prev, away_id),
        "home_gf_avg":    _gf_avg(home_prev, home_id),
        "home_gc_avg":    _gc_avg(home_prev, home_id),
        "away_gf_avg":    _gf_avg(away_prev, away_id),
        "away_gc_avg":    _gc_avg(away_prev, away_id),
        "home_position":  None,  # se une desde standings snapshot
        "away_position":  None,
        "h2h_home_wins":  int((h2h["result"] == "home").sum()) if len(h2h) else 0,
        "h2h_draws":      int((h2h["result"] == "draw").sum()) if len(h2h) else 0,
        "h2h_away_wins":  int((h2h["result"] == "away").sum()) if len(h2h) else 0,
    }


def _form_pts(prev: pd.DataFrame, team_id: int) -> float:
    if prev.empty:
        return 0.0
    pts = []
    for _, r in prev.iterrows():
        side = "home" if r["home_team_id"] == team_id else "away"
        pts.append(_points(r["home_score"], r["away_score"], side))
    return float(np.mean(pts)) if pts else 0.0


def _gf_avg(prev: pd.DataFrame, team_id: int) -> float:
    if prev.empty:
        return 0.0
    gf = []
    for _, r in prev.iterrows():
        if r["home_team_id"] == team_id:
            gf.append(r["home_score"] or 0)
        else:
            gf.append(r["away_score"] or 0)
    return float(np.mean(gf)) if gf else 0.0


def _gc_avg(prev: pd.DataFrame, team_id: int) -> float:
    if prev.empty:
        return 0.0
    gc = []
    for _, r in prev.iterrows():
        if r["home_team_id"] == team_id:
            gc.append(r["away_score"] or 0)
        else:
            gc.append(r["home_score"] or 0)
    return float(np.mean(gc)) if gc else 0.0
