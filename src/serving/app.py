"""API REST para predicciones de partidos LaLiga.

Endpoints:
  GET  /health                    -> {status: ok, model_loaded: bool}
  GET  /teams                     -> lista de equipos
  GET  /matches/upcoming          -> partidos proximos
  GET  /matches/by-jornada?jornada=N -> partidos de una jornada concreta
  POST /predict                   -> prediccion LightGBM (win/draw/loss)
  POST /predict/goals             -> prediccion Dixon-Coles (marcador)
  GET  /model/ratings             -> ratings ataque/defensa por equipo
  GET  /standings                 -> clasificacion actual
  POST /simulate/standings        -> simulacion Montecarlo de posicion final
"""
from __future__ import annotations

import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="LaLiga Analytics API",
    description="Predicciones de partidos LaLiga EA Sports",
    version="0.3.0",
)

_model_bundle: dict | None = None
_dc_model = None
_engine = None

# Equipos de Primera Division (clasificacion actual)
_PRIMERA_TEAMS_SUBQUERY = """
    SELECT team_id FROM standings
    WHERE fetched_at = (SELECT MAX(fetched_at) FROM standings)
"""


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    return _engine


def load_model():
    global _model_bundle
    if _model_bundle is None:
        path = Path("models/lgbm_v1.pkl")
        if path.exists():
            with open(path, "rb") as f:
                _model_bundle = pickle.load(f)
            logger.info("LightGBM model loaded from %s", path)
        else:
            logger.warning("Model file not found at %s", path)
    return _model_bundle


def load_dc_model():
    """Ajusta el modelo Dixon-Coles sobre el historico de Primera Division.

    El ajuste tarda ~5-15s en arranque y se cachea en memoria.
    """
    global _dc_model
    if _dc_model is None:
        with get_engine().connect() as conn:
            rows = conn.execute(text(f"""
                SELECT match_id, home_team_id, away_team_id,
                       home_score, away_score, kickoff_at
                FROM matches
                WHERE result IS NOT NULL
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                  AND competition_main = TRUE
                  AND home_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
                  AND away_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
                ORDER BY kickoff_at
            """)).mappings().all()

        if rows:
            df = pd.DataFrame([dict(r) for r in rows])
            df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True)
            from src.models.dixon_coles import DixonColesModel
            _dc_model = DixonColesModel().fit(df)
            logger.info("Dixon-Coles model ready (%d matches)", len(df))
        else:
            logger.warning("No historical matches available for Dixon-Coles")
    return _dc_model


@app.on_event("startup")
def startup():
    get_engine()
    load_model()
    # Dixon-Coles se carga en background para no bloquear el arranque
    import threading
    threading.Thread(target=load_dc_model, daemon=True).start()


# ── Schemas ─────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    season_id: Optional[int] = None


class PredictResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    prob_home: float
    prob_draw: float
    prob_away: float
    model: str


class SimulateRequest(BaseModel):
    team_id: int
    simulations: int = 5000


# ── Helpers ─────────────────────────────────────────────────────

def _predict_probs(home_id: int, away_id: int) -> tuple[float, float, float]:
    """Prediccion LightGBM: devuelve (prob_home, prob_draw, prob_away)."""
    bundle = load_model()
    if bundle is None:
        return (0.45, 0.28, 0.27)

    with get_engine().connect() as conn:
        feat_row = conn.execute(text("""
            SELECT f.*
            FROM match_features f
            JOIN matches m USING (match_id)
            WHERE m.home_team_id = :home AND m.away_team_id = :away
            ORDER BY m.kickoff_at DESC LIMIT 1
        """), {"home": home_id, "away": away_id}).mappings().first()

    feat_cols = bundle["feature_cols"]
    feat_values = {c: feat_row.get(c) if feat_row else None for c in feat_cols}

    X = pd.DataFrame([feat_values])[feat_cols]
    model = bundle["model"]

    from lightgbm import LGBMClassifier
    if not isinstance(model, LGBMClassifier):
        X = X.fillna(0)
    else:
        X = X.apply(pd.to_numeric, errors="coerce")

    probs_raw = model.predict_proba(X)[0]
    classes   = list(model.classes_)
    prob_map  = dict(zip(classes, probs_raw))
    return (
        float(prob_map.get("home", 0.45)),
        float(prob_map.get("draw", 0.28)),
        float(prob_map.get("away", 0.27)),
    )


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    load_model()
    return {
        "status": "ok",
        "lgbm_loaded": _model_bundle is not None,
        "dixon_coles_loaded": _dc_model is not None,
    }


@app.get("/teams")
def teams():
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT team_id, name, shortname FROM teams ORDER BY name"
        )).mappings().all()
    return {"teams": [dict(r) for r in rows]}


@app.get("/standings")
def standings():
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT s.position, t.name, t.shortname, t.team_id,
                   s.points, s.played, s.won, s.drawn, s.lost,
                   s.goals_for, s.goals_against, s.goal_difference, s.qualify_name
            FROM standings s
            JOIN teams t ON t.team_id = s.team_id
            WHERE s.fetched_at = (SELECT MAX(fetched_at) FROM standings)
            ORDER BY s.position
        """)).mappings().all()
    return {"standings": [dict(r) for r in rows]}


@app.get("/matches/upcoming")
def upcoming_matches(limit: int = 10):
    """Partidos proximos de Primera Division."""
    with get_engine().connect() as conn:
        rows = conn.execute(text(f"""
            SELECT m.match_id, m.kickoff_at, m.gameweek_week,
                   m.home_team_id, m.away_team_id,
                   ht.name AS home_team, at2.name AS away_team
            FROM matches m
            JOIN teams ht  ON ht.team_id  = m.home_team_id
            JOIN teams at2 ON at2.team_id = m.away_team_id
            WHERE m.result IS NULL
              AND m.competition_main = TRUE
              AND m.home_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
              AND m.away_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
            ORDER BY m.kickoff_at
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"matches": [dict(r) for r in rows]}


@app.get("/matches/by-jornada")
def matches_by_jornada(jornada: int):
    """Partidos de una jornada de Primera Division."""
    with get_engine().connect() as conn:
        rows = conn.execute(text(f"""
            SELECT m.match_id, m.kickoff_at, m.gameweek_week,
                   m.home_team_id, m.away_team_id,
                   ht.name AS home_team, at2.name AS away_team
            FROM matches m
            JOIN teams ht  ON ht.team_id  = m.home_team_id
            JOIN teams at2 ON at2.team_id = m.away_team_id
            WHERE m.gameweek_week = :jornada
              AND m.result IS NULL
              AND m.competition_main = TRUE
              AND m.home_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
              AND m.away_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
            ORDER BY m.kickoff_at
        """), {"jornada": jornada}).mappings().all()
    return {"matches": [dict(r) for r in rows]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Prediccion con LightGBM (clasificacion: home/draw/away)."""
    if load_model() is None:
        raise HTTPException(503, "LightGBM model not available")
    ph, pd_, pa = _predict_probs(req.home_team_id, req.away_team_id)
    return PredictResponse(
        home_team_id=req.home_team_id,
        away_team_id=req.away_team_id,
        prob_home=round(ph, 4),
        prob_draw=round(pd_, 4),
        prob_away=round(pa, 4),
        model="lgbm_v1",
    )


@app.post("/predict/goals")
def predict_goals(req: PredictRequest):
    """Prediccion de marcador con modelo Dixon-Coles.

    Devuelve:
    - lambda_home / lambda_away: goles esperados
    - prob_home / prob_draw / prob_away: probabilidades de resultado
    - most_likely_score: marcador mas probable (e.g. '1-0')
    - score_matrix: matriz 8x8 de probabilidades de marcador exacto
    """
    dc = load_dc_model()
    if dc is None:
        raise HTTPException(503, "Dixon-Coles model not available yet. Retry in a few seconds.")

    try:
        result = dc.predict(req.home_team_id, req.away_team_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    return {
        "home_team_id":      req.home_team_id,
        "away_team_id":      req.away_team_id,
        "model":             "dixon_coles_v1",
        **result,
    }


@app.get("/model/ratings")
def model_ratings():
    """Ratings de ataque y defensa por equipo (Dixon-Coles).

    Ataque alto -> equipo goleador.
    Defensa alta (menos negativa) -> equipo solido defensivamente.
    """
    dc = load_dc_model()
    if dc is None:
        raise HTTPException(503, "Dixon-Coles model not available yet.")

    with get_engine().connect() as conn:
        team_names = {r["team_id"]: r["name"] for r in conn.execute(text(
            "SELECT team_id, name FROM teams"
        )).mappings().all()}

    ratings = dc.team_ratings()
    ratings["name"] = ratings["team_id"].map(team_names).fillna("Unknown")
    ratings["attack"]  = ratings["attack"].round(4)
    ratings["defense"] = ratings["defense"].round(4)
    return {"ratings": ratings[["team_id", "name", "attack", "defense"]].to_dict(orient="records")}


@app.post("/simulate/standings")
def simulate_standings(req: SimulateRequest):
    """Simulacion Montecarlo de posicion final para un equipo de Primera.

    Campos de respuesta:
      pending_matches_count  -> partidos pendientes de Primera Division
      team_pending_count     -> partidos pendientes del equipo solicitado
      season_complete        -> True si no quedan partidos
    """
    if load_model() is None:
        raise HTTPException(503, "Model not available")

    with get_engine().connect() as conn:
        standing_rows = conn.execute(text("""
            SELECT t.team_id, s.points, s.goal_difference
            FROM standings s
            JOIN teams t ON t.team_id = s.team_id
            WHERE s.fetched_at = (SELECT MAX(fetched_at) FROM standings)
        """)).mappings().all()

        pending_rows = conn.execute(text(f"""
            SELECT m.match_id, m.home_team_id, m.away_team_id
            FROM matches m
            WHERE m.result IS NULL
              AND m.competition_main = TRUE
              AND m.home_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
              AND m.away_team_id IN ({_PRIMERA_TEAMS_SUBQUERY})
            ORDER BY m.kickoff_at
        """)).mappings().all()

    if not standing_rows:
        raise HTTPException(404, "No hay clasificacion disponible")

    pending_list  = list(pending_rows)
    pending_count = len(pending_list)

    team_pending_count = sum(
        1 for m in pending_list
        if m["home_team_id"] == req.team_id or m["away_team_id"] == req.team_id
    )

    if pending_count == 0:
        base_points = {r["team_id"]: r["points"] for r in standing_rows}
        base_gd     = {r["team_id"]: r["goal_difference"] for r in standing_rows}
        all_teams   = list(base_points.keys())
        ranked = sorted(all_teams,
                        key=lambda t: (base_points.get(t, 0), base_gd.get(t, 0)),
                        reverse=True)
        pos = ranked.index(req.team_id) + 1 if req.team_id in ranked else 20
        return {
            "team_id":               req.team_id,
            "simulations":           0,
            "pending_matches_count": 0,
            "team_pending_count":    0,
            "season_complete":       True,
            "position_distribution": {str(pos): 1.0},
        }

    match_probs = {}
    for m in pending_list:
        ph, pd_, pa = _predict_probs(m["home_team_id"], m["away_team_id"])
        match_probs[m["match_id"]] = {
            "home_id": m["home_team_id"],
            "away_id": m["away_team_id"],
            "probs":   [ph, pd_, pa],
        }

    base_points = {r["team_id"]: r["points"] for r in standing_rows}
    base_gd     = {r["team_id"]: r["goal_difference"] for r in standing_rows}
    all_teams   = list(base_points.keys())

    position_counts: dict[int, int] = defaultdict(int)
    N = min(req.simulations, 20000)

    for _ in range(N):
        pts = dict(base_points)
        gd  = dict(base_gd)

        for mp in match_probs.values():
            ph, pd_, pa = mp["probs"]
            outcome = random.choices(["home", "draw", "away"], weights=[ph, pd_, pa])[0]
            h, a = mp["home_id"], mp["away_id"]
            if outcome == "home":
                pts[h] = pts.get(h, 0) + 3
                gd[h]  = gd.get(h, 0)  + 1
                gd[a]  = gd.get(a, 0)  - 1
            elif outcome == "draw":
                pts[h] = pts.get(h, 0) + 1
                pts[a] = pts.get(a, 0) + 1
            else:
                pts[a] = pts.get(a, 0) + 3
                gd[a]  = gd.get(a, 0)  + 1
                gd[h]  = gd.get(h, 0)  - 1

        ranked = sorted(all_teams,
                        key=lambda t: (pts.get(t, 0), gd.get(t, 0)),
                        reverse=True)
        pos = ranked.index(req.team_id) + 1 if req.team_id in ranked else 20
        position_counts[pos] += 1

    distribution = {str(pos): round(cnt / N, 4)
                    for pos, cnt in sorted(position_counts.items())}

    return {
        "team_id":               req.team_id,
        "simulations":           N,
        "pending_matches_count": pending_count,
        "team_pending_count":    team_pending_count,
        "season_complete":       False,
        "position_distribution": distribution,
    }
