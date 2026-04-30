"""Feature store v2 con standings computados desde resultados (sin datos externos).

Todas las features son calculadas con datos disponibles
EXCLUSIVAMENTE antes del kickoff del partido objetivo.

Familias de features (22 activas):
  A - Estado competitivo (standings prepartido desde historial):
      home/away_points_total, home/away_table_position,
      position_diff, home/away_gd_total
  B - Forma reciente (ultimos 5 partidos de cualquier campo):
      home/away_goals_for/against_last5
  D - ELO dinamico:
      home_elo, away_elo, elo_diff
  E - Contexto:
      gameweek, home/away_rest_days, home/away_pressure_index
  F - Head-to-Head (ultimos 10 enfrentamientos directos):
      h2h_home_wins, h2h_draws, h2h_away_wins
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.standings_builder import build_match_standings

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v2.1.0"

_HOME_SCORE = "home_score"
_AWAY_SCORE = "away_score"
_FINISHED_STATUSES = {"FinishedPeriod", "FullTime", "Finished"}
H2H_WINDOW = 10


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
        s_home, s_away = (1.0, 0.0) if home_score > away_score else \
                         (0.0, 1.0) if home_score < away_score else (0.5, 0.5)
        self.ratings[home_id] = self._get(home_id) + self.k * (s_home - p_home)
        self.ratings[away_id] = self._get(away_id) + self.k * (s_away - p_away)

    def snapshot(self) -> Dict[int, float]:
        return dict(self.ratings)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Construye el dataset de features prepartido libre de leakage (22 features).

    Los standings se computan internamente desde el historial de resultados.
    No se requieren datos externos.
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
        self._standings_snap: Dict[int, Dict] = {}

    def build(self) -> pd.DataFrame:
        """Devuelve DataFrame con una fila por partido y todas las features."""
        self.matches = self.matches.sort_values("kickoff_at").reset_index(drop=True)
        self.matches["kickoff_at"] = pd.to_datetime(self.matches["kickoff_at"], utc=True)

        if self.standings.empty:
            snap_df = build_match_standings(self.matches)
            self._standings_snap = snap_df.set_index("match_id").to_dict(orient="index")

        rows: List[Dict] = []
        for _, match in self.matches.iterrows():
            match_id = int(match["match_id"])
            home_id  = int(match["home_team_id"])
            away_id  = int(match["away_team_id"])

            feats: Dict = {
                "match_id":     match_id,
                "season_id":    match.get("season_id"),
                "gameweek":     match.get("gameweek_week", -1),
                "kickoff_at":   match.get("kickoff_at"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_elo":     self._elo._get(home_id),
                "away_elo":     self._elo._get(away_id),
                "elo_diff":     self._elo._get(home_id) - self._elo._get(away_id),
            }
            feats.update(self._standing_features(match_id, home_id, away_id, int(match.get("gameweek_week", 1) or 1)))
            feats.update(self._form_features(home_id, away_id, match_id))
            feats.update(self._rest_features(home_id, away_id, match["kickoff_at"]))
            feats.update(self._h2h_features(home_id, away_id, match_id))
            rows.append(feats)

            home_score = match.get(_HOME_SCORE)
            away_score = match.get(_AWAY_SCORE)
            if pd.notna(home_score) and pd.notna(away_score):
                self._elo.update(home_id, away_id, int(home_score), int(away_score))

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _standing_features(self, match_id: int, home_id: int, away_id: int, gw: int) -> Dict:
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
        return {k: np.nan for k in [
            "home_points_total", "away_points_total",
            "home_table_position", "away_table_position", "position_diff",
            "home_gd_total", "away_gd_total",
            "home_pressure_index", "away_pressure_index",
        ]}

    def _form_features(self, home_id: int, away_id: int, current_match_id: int) -> Dict:
        """Forma reciente: goles en los ultimos 5 partidos de CUALQUIER campo.

        Corrige el bug anterior que solo contabilizaba partidos en casa para
        el local y partidos fuera para el visitante.
        """
        played = self.matches[
            (self.matches["match_id"] < current_match_id) &
            (self.matches["status"].isin(_FINISHED_STATUSES))
        ]

        def goals_for(team_id: int, n: int = 5) -> float:
            games = played[
                (played["home_team_id"] == team_id) | (played["away_team_id"] == team_id)
            ].tail(n)
            if games.empty:
                return np.nan
            is_home = games["home_team_id"] == team_id
            gf = np.where(is_home, games[_HOME_SCORE].fillna(0), games[_AWAY_SCORE].fillna(0))
            return float(gf.mean())

        def goals_against(team_id: int, n: int = 5) -> float:
            games = played[
                (played["home_team_id"] == team_id) | (played["away_team_id"] == team_id)
            ].tail(n)
            if games.empty:
                return np.nan
            is_home = games["home_team_id"] == team_id
            ga = np.where(is_home, games[_AWAY_SCORE].fillna(0), games[_HOME_SCORE].fillna(0))
            return float(ga.mean())

        return {
            "home_goals_for_last5":     goals_for(home_id),
            "home_goals_against_last5": goals_against(home_id),
            "away_goals_for_last5":     goals_for(away_id),
            "away_goals_against_last5": goals_against(away_id),
        }

    def _rest_features(self, home_id: int, away_id: int, kickoff: pd.Timestamp) -> Dict:
        """Dias de descanso desde el ultimo partido ANTES del kickoff actual."""
        kickoff_ts = pd.to_datetime(kickoff, utc=True)
        played = self.matches[
            self.matches["status"].isin(_FINISHED_STATUSES) &
            (pd.to_datetime(self.matches["kickoff_at"], utc=True) < kickoff_ts)
        ]

        def rest_days(team_id: int) -> float:
            m = played[(played["home_team_id"] == team_id) | (played["away_team_id"] == team_id)]
            if m.empty:
                return np.nan
            last = pd.to_datetime(m["kickoff_at"], utc=True).max()
            return float((kickoff_ts - last).days) if pd.notna(last) else np.nan

        return {"home_rest_days": rest_days(home_id), "away_rest_days": rest_days(away_id)}

    def _h2h_features(self, home_id: int, away_id: int, current_match_id: int) -> Dict:
        """Historial directo: ultimos H2H_WINDOW enfrentamientos entre ambos equipos.

        Perspectiva relativa al partido actual: home_id siempre es el 'local'.
        """
        h2h = self.matches[
            (self.matches["match_id"] < current_match_id) &
            (self.matches["status"].isin(_FINISHED_STATUSES)) &
            (
                ((self.matches["home_team_id"] == home_id) & (self.matches["away_team_id"] == away_id)) |
                ((self.matches["home_team_id"] == away_id) & (self.matches["away_team_id"] == home_id))
            )
        ].tail(H2H_WINDOW)

        if h2h.empty:
            return {"h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0}

        home_wins = away_wins = draws = 0
        for _, r in h2h.iterrows():
            result = r.get("result")
            if r["home_team_id"] == home_id:
                # home_id jugo como local en ese partido
                if result == "home":  draws_ = False; home_wins += 1
                elif result == "draw": draws += 1
                else: away_wins += 1
            else:
                # home_id jugo como visitante en ese partido
                if result == "away":  home_wins += 1
                elif result == "draw": draws += 1
                else: away_wins += 1

        return {"h2h_home_wins": home_wins, "h2h_draws": draws, "h2h_away_wins": away_wins}


# ---------------------------------------------------------------------------
# Columnas de features para entrenamiento (22 activas)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    # Familia D: ELO
    "home_elo", "away_elo", "elo_diff",
    # Familia A: Estado competitivo
    "home_points_total", "away_points_total",
    "home_table_position", "away_table_position", "position_diff",
    "home_gd_total", "away_gd_total",
    # Familia B: Forma reciente (todos los campos)
    "home_goals_for_last5", "home_goals_against_last5",
    "away_goals_for_last5", "away_goals_against_last5",
    # Familia E: Contexto
    "home_rest_days", "away_rest_days",
    "home_pressure_index", "away_pressure_index",
    "gameweek",
    # Familia F: Head-to-Head
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
]
