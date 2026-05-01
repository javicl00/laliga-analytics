"""Modelo Dixon-Coles (1997) para prediccion de marcadores en futbol.

Estima ratings de ataque (alpha) y defensa (beta) por equipo mediante
maximizacion de verosimilitud con penalizacion temporal (time-decay).
Incluye la correccion rho para partidos de pocos goles.

Referencia:
    Dixon, M. J., & Coles, S. G. (1997). Modelling association football
    scores and inefficiencies in the football betting market.
    Journal of the Royal Statistical Society: Series C, 46(2), 265-280.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Limites para parametros alpha/beta en log-escala.
# exp(4) ~ 54 goles esperados — imposible en futbol real.
# Evita que SLSQP diverja hacia +/-Inf en equipos con pocos partidos.
_PARAM_BOUND = 4.0

# Rango seguro para lambda/mu (goles esperados por partido).
# Por encima de 15 la PMF de Poisson es numericamente irrelevante.
_LAMBDA_MIN = 0.01
_LAMBDA_MAX = 15.0


def _safe_float(v: float) -> float:
    """Convierte NaN/Inf a 0.0 para garantizar serializacion JSON segura."""
    if math.isfinite(v):
        return v
    return 0.0


class DixonColesModel:
    """Modelo bivariate Poisson con correccion Dixon-Coles.

    Parametros por equipo:
        alpha_i  : factor de ataque (en log-escala)
        beta_i   : factor de defensa (en log-escala, valores altos = mejor defensa)

    Parametros globales:
        gamma    : ventaja de campo (log-escala)
        rho      : correccion para marcadores bajos (0-0, 1-0, 0-1, 1-1)

    Probabilidad de marcador:
        lambda = exp(alpha_home + beta_away + gamma)  <- goles esperados local
        mu     = exp(alpha_away + beta_home)          <- goles esperados visitante
        P(X=x, Y=y) = tau(x,y) * Poisson(x;lambda) * Poisson(y;mu)
    """

    def __init__(self, xi: float = 0.0018) -> None:
        """
        Args:
            xi: tasa de decaimiento temporal (por dia). 0.0018 ~> partidos
                de hace 1 anio tienen peso ~0.52 respecto a los recientes.
        """
        self.xi = xi
        self.params_: Optional[np.ndarray] = None
        self.teams_: List[int] = []
        self.team_idx_: Dict[int, int] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------
    # Correccion rho (Dixon-Coles)
    # ------------------------------------------------------------------

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """Factor de correccion para marcadores bajos."""
        if x == 0 and y == 0:
            return max(1.0 - lam * mu * rho, 1e-10)
        if x == 1 and y == 0:
            return 1.0 + mu * rho
        if x == 0 and y == 1:
            return 1.0 + lam * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    # ------------------------------------------------------------------
    # Log-verosimilitud
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        df: pd.DataFrame,
    ) -> float:
        n = len(self.teams_)
        alpha  = params[:n]
        beta   = params[n:2 * n]
        gamma  = params[2 * n]
        rho    = params[2 * n + 1]

        ll = 0.0
        for row in df.itertuples(index=False):
            hi = self.team_idx_[row.home_team_id]
            ai = self.team_idx_[row.away_team_id]
            x  = int(row.home_score)
            y  = int(row.away_score)
            w  = float(row.weight)

            lam = np.exp(alpha[hi] + beta[ai] + gamma)
            mu  = np.exp(alpha[ai] + beta[hi])

            tau = self._tau(x, y, lam, mu, rho)
            ll += w * (
                np.log(tau)
                + poisson.logpmf(x, lam)
                + poisson.logpmf(y, mu)
            )

        return -ll

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------

    def fit(self, matches_df: pd.DataFrame) -> "DixonColesModel":
        """Ajusta el modelo sobre el historico de partidos.

        Args:
            matches_df: DataFrame con columnas
                home_team_id, away_team_id, home_score, away_score, kickoff_at.
        """
        df = matches_df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True)

        # Pesos por decaimiento temporal
        max_date = df["kickoff_at"].max()
        df["days_ago"] = (max_date - df["kickoff_at"]).dt.days
        df["weight"] = np.exp(-self.xi * df["days_ago"])

        teams = sorted(
            set(df["home_team_id"].tolist() + df["away_team_id"].tolist())
        )
        self.teams_ = teams
        self.team_idx_ = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Params: [alpha_0..n, beta_0..n, gamma, rho]
        x0 = np.zeros(2 * n + 2)
        x0[2 * n]     =  0.3   # gamma: ventaja de campo
        x0[2 * n + 1] = -0.1   # rho

        # Bounds: alpha y beta acotados a [-_PARAM_BOUND, +_PARAM_BOUND]
        # para evitar divergencia numerica en equipos con pocos partidos.
        # gamma libre; rho acotado a (-1, 0] segun teoria Dixon-Coles.
        param_bounds = (
            [(-_PARAM_BOUND, _PARAM_BOUND)] * (2 * n)  # alpha + beta
            + [(None, None)]                             # gamma
            + [(-1.0, 0.0)]                              # rho
        )

        # Restriccion de identificabilidad: sum(alpha) = 0
        constraints = [{"type": "eq", "fun": lambda p, n=n: np.sum(p[:n])}]

        logger.info("Ajustando Dixon-Coles sobre %d partidos...", len(df))
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(df[["home_team_id", "away_team_id", "home_score", "away_score", "weight"]],),
            method="SLSQP",
            bounds=param_bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning(
                "Dixon-Coles no convergio completamente: %s "
                "(los parametros pueden ser suboptimos pero son finitos)",
                result.message,
            )

        self.params_ = result.x
        self.fitted_ = True
        logger.info(
            "Dixon-Coles ajustado. gamma=%.3f rho=%.3f convergencia=%s",
            result.x[2 * n], result.x[2 * n + 1], result.success,
        )
        return self

    # ------------------------------------------------------------------
    # Prediccion
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        max_goals: int = 8,
    ) -> dict:
        """Calcula la distribucion de probabilidad del marcador.

        Returns:
            dict con:
            - lambda_home, lambda_away: goles esperados
            - prob_home, prob_draw, prob_away
            - most_likely_score: str en formato 'X-Y'
            - score_matrix: lista 2D [home_goals][away_goals]
        """
        if not self.fitted_:
            raise RuntimeError("Modelo no ajustado. Llama a fit() primero.")

        hi = self.team_idx_.get(home_team_id)
        ai = self.team_idx_.get(away_team_id)
        if hi is None:
            raise ValueError(f"Equipo desconocido: {home_team_id}")
        if ai is None:
            raise ValueError(f"Equipo desconocido: {away_team_id}")

        n = len(self.teams_)
        alpha = self.params_[:n]
        beta  = self.params_[n:2 * n]
        gamma = self.params_[2 * n]
        rho   = self.params_[2 * n + 1]

        # Clamp lambda/mu para garantizar valores finitos en la PMF de Poisson.
        # Los bounds en fit() hacen que esto sea raro, pero es una red de seguridad.
        lam = float(np.clip(np.exp(alpha[hi] + beta[ai] + gamma), _LAMBDA_MIN, _LAMBDA_MAX))
        mu  = float(np.clip(np.exp(alpha[ai] + beta[hi]),          _LAMBDA_MIN, _LAMBDA_MAX))

        # Matriz de probabilidad de marcadores
        matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                tau = self._tau(x, y, lam, mu, rho)
                matrix[x, y] = tau * poisson.pmf(x, lam) * poisson.pmf(y, mu)

        # Sustituir cualquier NaN/Inf residual por 0 antes de renormalizar
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Renormalizacion por truncamiento
        total = matrix.sum()
        if total > 0:
            matrix /= total

        prob_home = _safe_float(float(np.sum(np.tril(matrix, -1))))
        prob_draw = _safe_float(float(np.sum(np.diag(matrix))))
        prob_away = _safe_float(float(np.sum(np.triu(matrix, 1))))

        best = np.unravel_index(matrix.argmax(), matrix.shape)

        return {
            "lambda_home":       round(lam, 4),
            "lambda_away":       round(mu, 4),
            "prob_home":         round(prob_home, 4),
            "prob_draw":         round(prob_draw, 4),
            "prob_away":         round(prob_away, 4),
            "most_likely_score": f"{best[0]}-{best[1]}",
            "score_matrix":      [
                [round(_safe_float(v), 6) for v in row]
                for row in matrix.tolist()
            ],
        }

    # ------------------------------------------------------------------
    # Ratings de equipos (exportable a UI)
    # ------------------------------------------------------------------

    def team_ratings(self) -> pd.DataFrame:
        """Devuelve DataFrame con attack/defense rating por equipo.

        Los valores NaN/Inf (posible si SLSQP no converge) se sustituyen
        por 0.0 para garantizar serializacion JSON segura en la API.
        """
        if not self.fitted_:
            raise RuntimeError("Modelo no ajustado")
        n = len(self.teams_)
        attack  = [_safe_float(v) for v in self.params_[:n].tolist()]
        defense = [_safe_float(v) for v in self.params_[n:2 * n].tolist()]
        return pd.DataFrame({
            "team_id": self.teams_,
            "attack":  attack,
            "defense": defense,
        }).sort_values("attack", ascending=False)
