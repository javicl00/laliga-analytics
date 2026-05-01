"""Modelo Dixon-Coles (1997) para prediccion de marcadores en futbol.

Estima ratings de ataque (alpha) y defensa (beta) por equipo mediante
maximizacion de verosimilitud con penalizacion temporal (time-decay).
Incluye la correccion rho para partidos de pocos goles.

Estrategia de identificabilidad:
    Se fija alpha[0] = 0 (equipo ancla) y se optimizan los n-1 alfas
    restantes. Esto es equivalente a sum(alpha)=0 pero evita conflictos
    entre la restriccion de igualdad y los bounds en SLSQP.

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
# exp(4) ~ 54 goles esperados: imposible en futbol real.
_PARAM_BOUND = 4.0

# Rango seguro para lambda/mu (goles esperados por partido).
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
        alpha_i  : factor de ataque (en log-escala); alpha[0] fijado a 0 (ancla)
        beta_i   : factor de defensa (en log-escala, valores altos = mejor defensa)

    Parametros globales:
        gamma    : ventaja de campo (log-escala)
        rho      : correccion para marcadores bajos (0-0, 1-0, 0-1, 1-1)

    Vector de optimizacion (tamano 2n+1):
        [ alpha_1..alpha_{n-1},  beta_0..beta_{n-1},  gamma,  rho ]
        alpha_0 = 0 fijo (no entra en el vector).

    Probabilidad de marcador:
        lambda = exp(alpha_home + beta_away + gamma)  <- goles esperados local
        mu     = exp(alpha_away + beta_home)          <- goles esperados visitante
        P(X=x, Y=y) = tau(x,y) * Poisson(x;lambda) * Poisson(y;mu)
    """

    def __init__(self, xi: float = 0.0018) -> None:
        self.xi = xi
        self.params_: Optional[np.ndarray] = None  # alpha completo (n valores)
        self.beta_: Optional[np.ndarray] = None
        self.gamma_: float = 0.0
        self.rho_: float = 0.0
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
    # Log-verosimilitud (opera sobre vector reducido)
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self,
        params_free: np.ndarray,
        df: pd.DataFrame,
        n: int,
    ) -> float:
        """params_free = [alpha_1..alpha_{n-1}, beta_0..beta_{n-1}, gamma, rho]"""
        # Reconstruir alpha completo: alpha[0]=0, resto libre
        alpha = np.concatenate([[0.0], params_free[: n - 1]])
        beta  = params_free[n - 1 : 2 * n - 1]
        gamma = params_free[2 * n - 1]
        rho   = params_free[2 * n]

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

        # Vector libre: [alpha_1..alpha_{n-1}, beta_0..beta_{n-1}, gamma, rho]
        # Tamano: (n-1) + n + 1 + 1 = 2n+1
        x0 = np.zeros(2 * n + 1)
        x0[2 * n - 1] =  0.3   # gamma
        x0[2 * n]     = -0.1   # rho

        # Bounds:
        #   alpha_1..alpha_{n-1} : [-_PARAM_BOUND, +_PARAM_BOUND]
        #   beta_0..beta_{n-1}   : [-_PARAM_BOUND, +_PARAM_BOUND]
        #   gamma                : libre
        #   rho                  : [-1.0, 0.0]
        bounds = (
            [(-_PARAM_BOUND, _PARAM_BOUND)] * (2 * n - 1)  # n-1 alphas + n betas
            + [(None, None)]                                  # gamma
            + [(-1.0, 0.0)]                                   # rho
        )

        logger.info("Ajustando Dixon-Coles sobre %d partidos...", len(df))
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(
                df[["home_team_id", "away_team_id", "home_score", "away_score", "weight"]],
                n,
            ),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning(
                "Dixon-Coles no convergio completamente: %s "
                "(parametros suboptimos pero finitos)",
                result.message,
            )

        # Almacenar parametros en forma completa para predict() y team_ratings()
        alpha_full = np.concatenate([[0.0], result.x[: n - 1]])
        beta_full  = result.x[n - 1 : 2 * n - 1]

        self.params_ = alpha_full          # alpha[0..n-1]
        self.beta_   = beta_full           # beta[0..n-1]
        self.gamma_  = float(result.x[2 * n - 1])
        self.rho_    = float(result.x[2 * n])
        self.fitted_ = True

        logger.info(
            "Dixon-Coles ajustado. gamma=%.3f rho=%.3f convergencia=%s",
            self.gamma_, self.rho_, result.success,
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
        """Calcula la distribucion de probabilidad del marcador."""
        if not self.fitted_:
            raise RuntimeError("Modelo no ajustado. Llama a fit() primero.")

        hi = self.team_idx_.get(home_team_id)
        ai = self.team_idx_.get(away_team_id)
        if hi is None:
            raise ValueError(f"Equipo desconocido: {home_team_id}")
        if ai is None:
            raise ValueError(f"Equipo desconocido: {away_team_id}")

        lam = float(np.clip(
            np.exp(self.params_[hi] + self.beta_[ai] + self.gamma_),
            _LAMBDA_MIN, _LAMBDA_MAX,
        ))
        mu = float(np.clip(
            np.exp(self.params_[ai] + self.beta_[hi]),
            _LAMBDA_MIN, _LAMBDA_MAX,
        ))

        matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                tau = self._tau(x, y, lam, mu, self.rho_)
                matrix[x, y] = tau * poisson.pmf(x, lam) * poisson.pmf(y, mu)

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

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
        """Devuelve DataFrame con attack/defense rating por equipo."""
        if not self.fitted_:
            raise RuntimeError("Modelo no ajustado")
        attack  = [_safe_float(v) for v in self.params_.tolist()]
        defense = [_safe_float(v) for v in self.beta_.tolist()]
        return pd.DataFrame({
            "team_id": self.teams_,
            "attack":  attack,
            "defense": defense,
        }).sort_values("attack", ascending=False)
