"""Simulador de clasificación final de temporada por Monte Carlo.

Usa las probabilidades partido a partido del modelo 1X2/goles
para simular el resto de la temporada N veces y obtener
distribuciones de posición, puntos, descenso y plazas europeas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    n_simulations: int = 10_000
    top_europe: int = 4      # Plazas Champions + Europa
    relegation_zone: int = 3  # Últimos descendidos
    tiebreak_gd: bool = True  # Desempate por diferencia de goles
    random_seed: int = 42


@dataclass
class TeamSimResult:
    team_id: int
    name: str
    mean_points: float
    std_points: float
    mean_position: float
    prob_champion: float
    prob_top4: float
    prob_relegation: float
    position_distribution: Dict[int, float] = field(repr=False)


class SeasonSimulator:
    """Simula el resto de la temporada por Monte Carlo.

    Parameters
    ----------
    config:
        Parámetros de simulación.
    predictor:
        Objeto con método predict_proba_1x2(X) -> (N, 3) y
        predict_goals(X) -> (lambda_home, lambda_away).
    """

    def __init__(
        self,
        predictor,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.predictor = predictor
        self.cfg = config or SimulationConfig()
        self._rng = np.random.default_rng(self.cfg.random_seed)

    def simulate(
        self,
        current_table: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        features_remaining: np.ndarray,
    ) -> List[TeamSimResult]:
        """Ejecuta N simulaciones del resto de la temporada.

        Parameters
        ----------
        current_table:
            DataFrame con columnas: team_id, name, points, goals_for, goals_against.
        remaining_fixtures:
            DataFrame con columnas: match_id, home_team_id, away_team_id.
        features_remaining:
            Array (M, F) de features prepartido para los partidos restantes.

        Returns
        -------
        Lista de TeamSimResult con estadísticas de simulación.
        """
        team_ids = current_table["team_id"].tolist()
        n_teams = len(team_ids)
        n_fixtures = len(remaining_fixtures)
        n_sim = self.cfg.n_simulations

        # Probabilidades 1X2 para todos los partidos restantes
        probs = self.predictor.predict_proba_1x2(features_remaining)  # (M, 3)
        lam_home, lam_away = self.predictor.predict_goals(features_remaining)  # (M,)

        # Acumuladores
        points_acc = np.zeros((n_sim, n_teams), dtype=np.int32)
        gf_acc = np.zeros((n_sim, n_teams), dtype=np.int32)
        ga_acc = np.zeros((n_sim, n_teams), dtype=np.int32)

        # Puntos de partida
        base_points = np.array([int(current_table.loc[current_table["team_id"] == t, "points"].iloc[0]) for t in team_ids])
        base_gf = np.array([int(current_table.loc[current_table["team_id"] == t, "goals_for"].iloc[0]) for t in team_ids])
        base_ga = np.array([int(current_table.loc[current_table["team_id"] == t, "goals_against"].iloc[0]) for t in team_ids])

        team_idx = {t: i for i, t in enumerate(team_ids)}

        logger.info("Starting %d simulations for %d remaining fixtures", n_sim, n_fixtures)

        for sim in range(n_sim):
            pts = base_points.copy()
            gf = base_gf.copy()
            ga = base_ga.copy()

            for fi, (_, fix) in enumerate(remaining_fixtures.iterrows()):
                h_idx = team_idx.get(fix["home_team_id"])
                a_idx = team_idx.get(fix["away_team_id"])
                if h_idx is None or a_idx is None:
                    continue

                # Simular resultado
                p_h, p_d, p_a = probs[fi]
                outcome = self._rng.choice([0, 1, 2], p=[p_h, p_d, p_a])

                # Simular goles desde Poisson
                h_goals = int(self._rng.poisson(lam_home[fi]))
                a_goals = int(self._rng.poisson(lam_away[fi]))

                # Ajustar goles para que sean coherentes con el resultado
                if outcome == 0 and h_goals <= a_goals:
                    h_goals = a_goals + 1
                elif outcome == 2 and a_goals <= h_goals:
                    a_goals = h_goals + 1
                elif outcome == 1:
                    a_goals = h_goals

                gf[h_idx] += h_goals; ga[h_idx] += a_goals
                gf[a_idx] += a_goals; ga[a_idx] += h_goals

                if outcome == 0:
                    pts[h_idx] += 3
                elif outcome == 2:
                    pts[a_idx] += 3
                else:
                    pts[h_idx] += 1; pts[a_idx] += 1

            points_acc[sim] = pts
            gf_acc[sim] = gf
            ga_acc[sim] = ga

        return self._aggregate(team_ids, current_table, points_acc, gf_acc, ga_acc, n_sim)

    def _aggregate(
        self,
        team_ids: List[int],
        current_table: pd.DataFrame,
        points_acc: np.ndarray,
        gf_acc: np.ndarray,
        ga_acc: np.ndarray,
        n_sim: int,
    ) -> List[TeamSimResult]:
        n_teams = len(team_ids)
        position_counts = np.zeros((n_teams, n_teams), dtype=np.int32)

        for sim in range(n_sim):
            gd = gf_acc[sim] - ga_acc[sim]
            order = np.lexsort((-gd, -points_acc[sim]))  # puntos desc, gd desc
            for rank, idx in enumerate(order):
                position_counts[idx, rank] += 1

        results = []
        for i, team_id in enumerate(team_ids):
            name_rows = current_table[current_table["team_id"] == team_id]["name"]
            name = str(name_rows.iloc[0]) if not name_rows.empty else str(team_id)
            results.append(TeamSimResult(
                team_id=team_id,
                name=name,
                mean_points=float(points_acc[:, i].mean()),
                std_points=float(points_acc[:, i].std()),
                mean_position=float(np.average(np.arange(1, n_teams + 1), weights=position_counts[i])),
                prob_champion=float(position_counts[i, 0] / n_sim),
                prob_top4=float(position_counts[i, :self.cfg.top_europe].sum() / n_sim),
                prob_relegation=float(position_counts[i, -self.cfg.relegation_zone:].sum() / n_sim),
                position_distribution={int(p + 1): float(position_counts[i, p] / n_sim) for p in range(n_teams)},
            ))

        results.sort(key=lambda r: r.mean_position)
        return results
