"""Feature store v2 con standings computados desde resultados (sin datos externos).

Todas las features son calculadas con datos disponibles
EXCLUSIVAMENTE antes del kickoff del partido objetivo.

Familias de features:
  A - Estado competitivo (standings snapshot prepartido):
      home/away_points_total, home/away_table_position,
      position_diff, home/away_gd_total
  B - Forma reciente (ultimos 5 partidos):
      home/away_goals_for/against_last5
  D - ELO dinamico:
      home_elo, away_elo, elo_diff
  E - Contexto:
      gameweek, home/away_rest_days, home/away_pressure_index

Esquema DB real (tabla matches):
  scores  -> home_score / away_score  (smallint, nullable)
  result  -> 'home' | 'draw' | 'away'
  status  -> text (FullTime, FinishedPeriod, etc.)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.standings_builder import build_match_standings

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v2.0.0"

# ---------------------------------------------------------------------------
# Columnas de scores en la tabla matches
# ---------------------------------------------------------------------------
_HOME_SCORE = "home_score"
_AWAY_SCORE = "away_score"
_FINISHED_STATUSES = {"FinishedPeriod", "FullTime", "Finished"}


# ---------------------------------------------------------------------------
# Ratings Elo
# ---------------------------------------------------------------------------

class EloRating:
    """Ratings Elo dinamicos por equipo con factores home/away."""

    def __init__(self, base: float = 1500.0, k: float = 32.0, home_advantage: float = 70.0) -> None:
        self.base = base
        self.k = k
        self.home_advantage = home_advantage
        self.ratings: Dict[int, float] = {}

    def _get(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.base)

    def expected(self, home_id: int, away_id: int) -> Tuple[float, float]:
        r_home = self._get(home_id) + self.home_advantage
        r_away = self._get(away_id)
        p_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        return p_home, 1.0 - p_home

    def update(self, home_id: int, away_id: int, home_score: int, away_score: int) -> None:
        p_home, p_away = self.expected(home_id, away_id)
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
        self.ratings[home_id] = self._get(home_id) + self.k * (s_home - p_home)
        self.ratings[away_id] = self._get(away_id) + self.k * (s_away - p_away)

    def snapshot(self) -> Dict[int, float]:
        return dict(self.ratings)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Construye el dataset de features prepartido libre de leakage.

    Los standings se computan internamente desde el historial de resultados
    via standings_builder.build_match_standings() cuando standings_df es None.
    No se requieren datos externos para las 17 features activas.

    Parameters
    ----------
    matches_df:
        DataFrame con columnas: match_id, season_id, gameweek_week, kickoff_at,
        home_team_id, away_team_id, home_score, away_score, result, status.
    standings_df:
        Opcional. DataFrame con snapshots por jornada desde API externa.
        Si se omite, los standings se reconstruyen desde matches_df.
    team_stats_df:
        Opcional. DataFrame con estadisticas de equipo (shots, possession, etc.).
    """

    def __init__(
        self,
        matches_df: pd.DataFrame,
        standings_df: Optional[pd.DataFrame] = None,
        team_stats_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.matches = matches_df.copy()
        self.standings = standings_df.copy() if standings_df is not None else pd.DataFrame()
        self.team_stats = team_stats_df.copy() if team_stats_df is not None else pd.DataFrame()
        self._elo = EloRating()
        # Standings snap: match_id -> {home_points_total, away_points_total, ...}
        self._standings_snap: Dict[int, Dict] = {}

    def build(self) -> pd.DataFrame:
        """Devuelve DataFrame con una fila por partido y todas las features."""
        self.matches = self.matches.sort_values("kickoff_at").reset_index(drop=True)
        self.matches["kickoff_at"] = pd.to_datetime(self.matches["kickoff_at"], utc=True)

        # Pre-computar standings desde resultados si no hay standings_df externo
        if self.standings.empty:
            snap_df = build_match_standings(self.matches)
            self._standings_snap = snap_df.set_index("match_id").to_dict(orient="index")
            logger.debug("Loaded %d standings snapshots from match history", len(self._standings_snap))

        rows: List[Dict] = []

        for _, match in self.matches.iterrows():
            match_id = int(match["match_id"])
            gw       = match.get("gameweek_week", -1)
            home_id  = int(match["home_team_id"])
            away_id  = int(match["away_team_id"])

            feats: Dict = {
                "match_id":     match_id,
                "season_id":    match.get("season_id"),
                "gameweek":     gw,
                "kickoff_at":   match.get("kickoff_at"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                # ELO antes del partido
                "home_elo":  self._elo._get(home_id),
                "away_elo":  self._elo._get(away_id),
                "elo_diff":  self._elo._get(home_id) - self._elo._get(away_id),
            }

            # Standings snapshot prepartido
            feats.update(self._standing_features(match_id, home_id, away_id, gw))

            # Forma reciente
            feats.update(self._form_features(home_id, away_id, match_id))

            # Descanso
            feats.update(self._rest_features(home_id, away_id, match["kickoff_at"]))

            rows.append(feats)

            # Actualizar Elo DESPUES de la fila (anti-leakage)
            home_score = match.get(_HOME_SCORE)
            away_score = match.get(_AWAY_SCORE)
            if pd.notna(home_score) and pd.notna(away_score):
                self._elo.update(home_id, away_id, int(home_score), int(away_score))

        return pd.DataFrame(rows)

    def _standing_features(self, match_id: int, home_id: int, away_id: int, gw: int) -> Dict:
        """Devuelve features de standings para el partido, desde el snap prepartido.

        Usa _standings_snap (computado desde matches) con fallback al
        standings_df externo por gameweek_id para compatibilidad.
        """
        # Camino primario: snap pre-computado por standings_builder
        if self._standings_snap:
            snap = self._standings_snap.get(match_id)
            if snap:
                return {
                    "home_points_total":   snap.get("home_points_total", np.nan),
                    "away_points_total":   snap.get("away_points_total", np.nan),
                    "home_table_position": snap.get("home_table_position", np.nan),
                    "away_table_position": snap.get("away_table_position", np.nan),
                    "position_diff":       snap.get("position_diff", np.nan),
                    "home_gd_total":       snap.get("home_gd_total", np.nan),
                    "away_gd_total":       snap.get("away_gd_total", np.nan),
                    "home_pressure_index": snap.get("home_pressure_index", np.nan),
                    "away_pressure_index": snap.get("away_pressure_index", np.nan),
                }

        # Fallback: standings_df externo por gameweek_id (API)
        null_feats = {k: np.nan for k in [
            "home_points_total", "away_points_total",
            "home_table_position", "away_table_position",
            "position_diff", "home_gd_total", "away_gd_total",
            "home_pressure_index", "away_pressure_index",
        ]}
        if self.standings.empty or gw < 2:
            return null_feats

        prev = self.standings[self.standings["gameweek_id"] == gw - 1]
        h = prev[prev["team_id"] == home_id]
        a = prev[prev["team_id"] == away_id]
        h_pts   = float(h["points"].iloc[0])       if not h.empty else np.nan
        a_pts   = float(a["points"].iloc[0])       if not a.empty else np.nan
        h_pos   = float(h["position"].iloc[0])     if not h.empty else np.nan
        a_pos   = float(a["position"].iloc[0])     if not a.empty else np.nan
        h_gf    = float(h["goals_for"].iloc[0])    if not h.empty else np.nan
        h_ga    = float(h["goals_against"].iloc[0]) if not h.empty else np.nan
        a_gf    = float(a["goals_for"].iloc[0])    if not a.empty else np.nan
        a_ga    = float(a["goals_against"].iloc[0]) if not a.empty else np.nan
        h_gd    = h_gf - h_ga if not np.isnan(h_gf) else np.nan
        a_gd    = a_gf - a_ga if not np.isnan(a_gf) else np.nan
        pos_diff = h_pos - a_pos if not np.isnan(h_pos) else np.nan
        remaining = max(0, 38 - gw)

        def pressure(pos: float) -> float:
            if np.isnan(pos):
                return np.nan
            p = int(pos)
            if p >= 16:
                return remaining / (remaining + 1.0) * (p / 20.0)
            elif p <= 4:
                return remaining / (remaining + 1.0) * (1.0 - p / 20.0)
            return 0.1

        return {
            "home_points_total":   h_pts,
            "away_points_total":   a_pts,
            "home_table_position": h_pos,
            "away_table_position": a_pos,
            "position_diff":       pos_diff,
            "home_gd_total":       h_gd,
            "away_gd_total":       a_gd,
            "home_pressure_index": pressure(h_pos),
            "away_pressure_index": pressure(a_pos),
        }

    def _form_features(self, home_id: int, away_id: int, current_match_id) -> Dict:
        """Rolling de goles en los ultimos 5 partidos (ANTES del actual)."""
        played = self.matches[
            (self.matches["match_id"] < current_match_id) &
            (self.matches["status"].isin(_FINISHED_STATUSES))
        ]

        def last_n_goals_for(team_id, as_home: bool, n: int = 5) -> float:
            if as_home:
                m = played[played["home_team_id"] == team_id].tail(n)
                return float(m[_HOME_SCORE].mean()) if not m.empty else np.nan
            else:
                m = played[played["away_team_id"] == team_id].tail(n)
                return float(m[_AWAY_SCORE].mean()) if not m.empty else np.nan

        def last_n_goals_against(team_id, as_home: bool, n: int = 5) -> float:
            if as_home:
                m = played[played["home_team_id"] == team_id].tail(n)
                return float(m[_AWAY_SCORE].mean()) if not m.empty else np.nan
            else:
                m = played[played["away_team_id"] == team_id].tail(n)
                return float(m[_HOME_SCORE].mean()) if not m.empty else np.nan

        return {
            "home_goals_for_last5":     last_n_goals_for(home_id, as_home=True),
            "home_goals_against_last5": last_n_goals_against(home_id, as_home=True),
            "away_goals_for_last5":     last_n_goals_for(away_id, as_home=False),
            "away_goals_against_last5": last_n_goals_against(away_id, as_home=False),
        }

    def _rest_features(self, home_id: int, away_id: int, kickoff: pd.Timestamp) -> Dict:
        """Dias de descanso desde el ultimo partido ANTES del kickoff actual."""
        kickoff_ts = pd.to_datetime(kickoff, utc=True)
        played = self.matches[
            self.matches["status"].isin(_FINISHED_STATUSES) &
            (pd.to_datetime(self.matches["kickoff_at"], utc=True) < kickoff_ts)
        ]

        def rest_days(team_id: int) -> float:
            team_matches = played[
                (played["home_team_id"] == team_id) | (played["away_team_id"] == team_id)
            ]
            if team_matches.empty:
                return np.nan
            last = pd.to_datetime(team_matches["kickoff_at"], utc=True).max()
            if pd.isna(last):
                return np.nan
            return float((kickoff_ts - last).days)

        return {
            "home_rest_days": rest_days(home_id),
            "away_rest_days": rest_days(away_id),
        }


# ---------------------------------------------------------------------------
# Columnas de features para entrenamiento (17 features activas)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    # Familia D: ELO
    "home_elo", "away_elo", "elo_diff",
    # Familia A: Estado competitivo
    "home_points_total", "away_points_total",
    "home_table_position", "away_table_position", "position_diff",
    "home_gd_total", "away_gd_total",
    # Familia B: Forma reciente
    "home_goals_for_last5", "home_goals_against_last5",
    "away_goals_for_last5", "away_goals_against_last5",
    # Familia E: Contexto
    "home_rest_days", "away_rest_days",
    "home_pressure_index", "away_pressure_index",
    "gameweek",
]
