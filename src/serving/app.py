"""API REST para predicciones de partidos LaLiga.

Endpoints:
  GET  /health                    -> {status: ok, model_loaded: bool}
  GET  /teams                     -> lista de equipos
  GET  /matches/upcoming          -> partidos proximos
  GET  /matches/by-jornada?jornada=N -> partidos de una jornada concreta
  POST /predict                   -> prediccion para un partido
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
    version="0.2.0",
)

_model_bundle: dict | None = None
_engine = None


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
            logger.info("Model loaded from %s", path)
        else:
            logger.warning("Model file not found at %s", path)
    return _model_bundle


@app.on_event("startup")
def startup():
    get_engine()
    load_model()


# ── Schemas ───────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────

def _predict_probs(home_id: int, away_id: int) -> tuple[float, float, float]:
    """Devuelve (prob_home, prob_draw, prob_away) para un partido."""
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


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    load_model()
    return {"status": "ok", "model_loaded": _model_bundle is not None}


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
    """Partidos proximos: filtra por result IS NULL."""
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT m.match_id, m.kickoff_at, m.gameweek_week,
                   m.home_team_id, m.away_team_id,
                   ht.name AS home_team, at2.name AS away_team
            FROM matches m
            JOIN teams ht  ON ht.team_id  = m.home_team_id
            JOIN teams at2 ON at2.team_id = m.away_team_id
            WHERE m.result IS NULL
              AND m.competition_main = TRUE
            ORDER BY m.kickoff_at
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"matches": [dict(r) for r in rows]}


@app.get("/matches/by-jornada")
def matches_by_jornada(jornada: int):
    """Partidos de una jornada: filtra por result IS NULL."""
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT m.match_id, m.kickoff_at, m.gameweek_week,
                   m.home_team_id, m.away_team_id,
                   ht.name AS home_team, at2.name AS away_team
            FROM matches m
            JOIN teams ht  ON ht.team_id  = m.home_team_id
            JOIN teams at2 ON at2.team_id = m.away_team_id
            WHERE m.gameweek_week = :jornada
              AND m.result IS NULL
              AND m.competition_main = TRUE
            ORDER BY m.kickoff_at
        """), {"jornada": jornada}).mappings().all()
    return {"matches": [dict(r) for r in rows]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if load_model() is None:
        raise HTTPException(503, "Model not available")
    ph, pd_, pa = _predict_probs(req.home_team_id, req.away_team_id)
    return PredictResponse(
        home_team_id=req.home_team_id,
        away_team_id=req.away_team_id,
        prob_home=round(ph, 4),
        prob_draw=round(pd_, 4),
        prob_away=round(pa, 4),
        model="lgbm_v1",
    )


@app.post("/simulate/standings")
def simulate_standings(req: SimulateRequest):
    """Simulacion Montecarlo: distribucion de posicion final para un equipo.

    Devuelve pending_matches_count para que el cliente sepa si la simulacion
    tiene varianza real o la temporada ya esta completada.
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

        pending_rows = conn.execute(text("""
            SELECT m.match_id, m.home_team_id, m.away_team_id
            FROM matches m
            WHERE m.result IS NULL
              AND m.competition_main = TRUE
            ORDER BY m.kickoff_at
        """)).mappings().all()

    if not standing_rows:
        raise HTTPException(404, "No hay clasificacion disponible")

    pending_list = list(pending_rows)
    pending_count = len(pending_list)

    # Sin partidos pendientes: clasificacion final ya determinada
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
            "season_complete":        True,
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
        "season_complete":        False,
        "position_distribution": distribution,
    }
