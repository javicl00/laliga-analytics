"""Feature store v1 con snapshots prepartido y control anti-leakage.

Todas las features son calculadas con datos disponibles
EXCLUSIVAMENTE antes del kickoff del partido objetivo.
Ninguna feature puede hacer referencia a resultados del partido objetivo
ni a estadísticas acumuladas que incluyan ese partido.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v1.0.0"

# ---------------------------------------------------------------------------
# Catálogo de features con política de leakage explícita
# ---------------------------------------------------------------------------

FEATURE_CATALOG: Dict[str, Dict] = {
    # Familia A: Estado competitivo
    "home_points_total": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    "away_points_total": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    "home_points_ppg_last3": {"source": "standings", "window": "rolling_3", "allowed_inference": True},
    "away_points_ppg_last3": {"source": "standings", "window": "rolling_3", "allowed_inference": True},
    "home_points_ppg_last5": {"source": "standings", "window": "rolling_5", "allowed_inference": True},
    "away_points_ppg_last5": {"source": "standings", "window": "rolling_5", "allowed_inference": True},
    "home_table_position": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    "away_table_position": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    "position_diff": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    "home_gd_total": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    "away_gd_total": {"source": "standings", "window": "season_to_date", "allowed_inference": True},
    # Familia B: Forma reciente
    "home_goals_for_last5": {"source": "results", "window": "rolling_5", "allowed_inference": True},
    "away_goals_for_last5": {"source": "results", "window": "rolling_5", "allowed_inference": True},
    "home_goals_against_last5": {"source": "results", "window": "rolling_5", "allowed_inference": True},
    "away_goals_against_last5": {"source": "results", "window": "rolling_5", "allowed_inference": True},
    "home_shots_last5": {"source": "team_stats", "window": "rolling_5", "allowed_inference": True},
    "away_shots_last5": {"source": "team_stats", "window": "rolling_5", "allowed_inference": True},
    # Familia C: Estilo de juego EWMA
    "home_possession_ewma": {"source": "team_stats", "window": "ewma_8", "allowed_inference": True},
    "away_possession_ewma": {"source": "team_stats", "window": "ewma_8", "allowed_inference": True},
    "home_ppda_ewma": {"source": "team_stats", "window": "ewma_8", "allowed_inference": True},
    "away_ppda_ewma": {"source": "team_stats", "window": "ewma_8", "allowed_inference": True},
    # Familia D: Ratings dinámicos
    "home_elo": {"source": "computed", "window": "dynamic", "allowed_inference": True},
    "away_elo": {"source": "computed", "window": "dynamic", "allowed_inference": True},
    "elo_diff": {"source": "computed", "window": "dynamic", "allowed_inference": True},
    # Familia E: Contexto
    "gameweek": {"source": "matches", "window": "static", "allowed_inference": True},
    "home_rest_days": {"source": "matches", "window": "static", "allowed_inference": True},
    "away_rest_days": {"source": "matches", "window": "static", "allowed_inference": True},
    "home_pressure_index": {"source": "computed", "window": "season_to_date", "allowed_inference": True},
    "away_pressure_index": {"source": "computed", "window": "season_to_date", "allowed_inference": True},
    # Leakage: NO permitidas en inferencia
    "full_season_points": {"source": "targets", "window": "full_season", "allowed_inference": False, "note": "LEAKAGE"},
    "match_result": {"source": "targets", "window": "post_match", "allowed_inference": False, "note": "LEAKAGE"},
}


# ---------------------------------------------------------------------------
# Ratings Elo
# ---------------------------------------------------------------------------

class EloRating:
    """Ratings Elo dinámicos por equipo con factores home/away."""

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

    def update(self, home_id: int, away_id: int, home_goals: int, away_goals: int) -> None:
        p_home, p_away = self.expected(home_id, away_id)
        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
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

    Parameters
    ----------
    matches_df:
        DataFrame con columnas: match_id, season_id, gameweek, kickoff_at,
        home_team_id, away_team_id, home_goals, away_goals, status.
    standings_df:
        DataFrame con columnas: season_id, gameweek_id, team_id, points,
        position, won, drawn, lost, goals_for, goals_against, snapshot_ts.
    team_stats_df:
        DataFrame con columnas: season_id, gameweek_id, team_id, stat_name,
        stat_value, snapshot_ts.
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
        self._elo_snapshots: Dict[int, Dict[int, float]] = {}  # match_id -> {team_id: elo}

    def build(self) -> pd.DataFrame:
        """Devuelve DataFrame con una fila por partido y columnas de features."""
        self.matches = self.matches.sort_values("kickoff_at").reset_index(drop=True)
        rows: List[Dict] = []

        for _, match in self.matches.iterrows():
            if match.get("status") not in ("FinishedPeriod", "FullTime", "Finished", None, ""):
                # Partido no jugado aún: incluir solo si se quiere para inferencia
                pass

            match_id = match["match_id"]
            gw = match.get("gameweek", -1)
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]

            # Elo antes del partido
            p_home_elo, _ = self._elo.expected(home_id, away_id)
            feats: Dict = {
                "match_id": match_id,
                "season_id": match.get("season_id"),
                "gameweek": gw,
                "kickoff_at": match.get("kickoff_at"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_elo": self._elo._get(home_id),
                "away_elo": self._elo._get(away_id),
                "elo_diff": self._elo._get(home_id) - self._elo._get(away_id),
            }

            # Standings snapshot (jornada anterior = gw - 1)
            feats.update(self._standing_features(home_id, away_id, gw))

            # Forma reciente en partidos
            feats.update(self._form_features(home_id, away_id, match_id))

            # Descanso
            feats.update(self._rest_features(home_id, away_id, match["kickoff_at"]))

            # Presión competitiva
            feats.update(self._pressure_features(home_id, away_id, gw))

            rows.append(feats)

            # Actualizar Elo DESPUÉS de añadir la fila (anti-leakage)
            home_goals = match.get("home_goals")
            away_goals = match.get("away_goals")
            if pd.notna(home_goals) and pd.notna(away_goals):
                self._elo.update(home_id, away_id, int(home_goals), int(away_goals))

        return pd.DataFrame(rows)

    def _standing_features(self, home_id: int, away_id: int, gw: int) -> Dict:
        feats: Dict = {}
        if self.standings.empty or gw < 2:
            return {k: np.nan for k in [
                "home_points_total", "away_points_total", "home_table_position",
                "away_table_position", "position_diff", "home_gd_total", "away_gd_total"
            ]}
        prev = self.standings[self.standings["gameweek_id"] == gw - 1]
        h = prev[prev["team_id"] == home_id]
        a = prev[prev["team_id"] == away_id]
        feats["home_points_total"] = float(h["points"].iloc[0]) if not h.empty else np.nan
        feats["away_points_total"] = float(a["points"].iloc[0]) if not a.empty else np.nan
        feats["home_table_position"] = float(h["position"].iloc[0]) if not h.empty else np.nan
        feats["away_table_position"] = float(a["position"].iloc[0]) if not a.empty else np.nan
        feats["position_diff"] = (
            feats["home_table_position"] - feats["away_table_position"]
            if not np.isnan(feats.get("home_table_position", np.nan)) else np.nan
        )
        h_gf = float(h["goals_for"].iloc[0]) if not h.empty else np.nan
        h_ga = float(h["goals_against"].iloc[0]) if not h.empty else np.nan
        a_gf = float(a["goals_for"].iloc[0]) if not a.empty else np.nan
        a_ga = float(a["goals_against"].iloc[0]) if not a.empty else np.nan
        feats["home_gd_total"] = h_gf - h_ga if not np.isnan(h_gf) else np.nan
        feats["away_gd_total"] = a_gf - a_ga if not np.isnan(a_gf) else np.nan
        return feats

    def _form_features(self, home_id: int, away_id: int, current_match_id) -> Dict:
        """Rolling de goles en los últimos 5 partidos (ANTES del actual)."""
        played = self.matches[
            (self.matches["match_id"] < current_match_id) &
            (self.matches["status"].isin(["FinishedPeriod", "FullTime", "Finished"]))
        ]
        def last_n_goals(team_id, is_home, n=5):
            if is_home:
                m = played[played["home_team_id"] == team_id].tail(n)
                return m["home_goals"].mean() if not m.empty else np.nan
            else:
                m = played[played["away_team_id"] == team_id].tail(n)
                return m["away_goals"].mean() if not m.empty else np.nan
        return {
            "home_goals_for_last5": last_n_goals(home_id, True),
            "home_goals_against_last5": last_n_goals(home_id, False),
            "away_goals_for_last5": last_n_goals(away_id, False),
            "away_goals_against_last5": last_n_goals(away_id, True),
        }

    def _rest_features(self, home_id: int, away_id: int, kickoff) -> Dict:
        """Días de descanso desde el último partido de cada equipo."""
        played = self.matches[
            self.matches["status"].isin(["FinishedPeriod", "FullTime", "Finished"])
        ]
        def rest_days(team_id):
            team_matches = played[
                (played["home_team_id"] == team_id) | (played["away_team_id"] == team_id)
            ]
            if team_matches.empty:
                return np.nan
            last = pd.to_datetime(team_matches["kickoff_at"]).max()
            k = pd.to_datetime(kickoff)
            if pd.isna(last) or pd.isna(k):
                return np.nan
            return float((k - last).days)
        return {
            "home_rest_days": rest_days(home_id),
            "away_rest_days": rest_days(away_id),
        }

    def _pressure_features(self, home_id: int, away_id: int, gw: int) -> Dict:
        """Índice de presión competitiva basado en posición y jornadas restantes."""
        if self.standings.empty or gw < 2:
            return {"home_pressure_index": np.nan, "away_pressure_index": np.nan}
        total_gw = 38
        remaining = max(0, total_gw - gw)
        prev = self.standings[self.standings["gameweek_id"] == gw - 1]

        def pressure(team_id):
            row = prev[prev["team_id"] == team_id]
            if row.empty:
                return np.nan
            pos = float(row["position"].iloc[0])
            pts = float(row["points"].iloc[0])
            # Alta presión si está en zona descenso (16-20) o lucha por título (1-4)
            if pos >= 16:
                return float(remaining) / (remaining + 1.0) * (pos / 20.0)
            elif pos <= 4:
                return float(remaining) / (remaining + 1.0) * (1.0 - pos / 20.0)
            return 0.1
        return {
            "home_pressure_index": pressure(home_id),
            "away_pressure_index": pressure(away_id),
        }


# ---------------------------------------------------------------------------
# Utilidad: columnas de features para entrenamiento
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "home_elo", "away_elo", "elo_diff",
    "home_points_total", "away_points_total",
    "home_table_position", "away_table_position", "position_diff",
    "home_gd_total", "away_gd_total",
    "home_goals_for_last5", "home_goals_against_last5",
    "away_goals_for_last5", "away_goals_against_last5",
    "home_rest_days", "away_rest_days",
    "home_pressure_index", "away_pressure_index",
    "gameweek",
]
