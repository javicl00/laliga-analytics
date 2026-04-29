"""API REST para predicciones de partidos LaLiga.

Endpoints:
  GET  /health               -> {status: ok}
  GET  /matches/upcoming     -> partidos proximos con predicciones
  POST /predict              -> prediccion para un partido concreto
  GET  /standings            -> clasificacion actual
"""
from __future__ import annotations

import logging
import os
import pickle
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
    version="0.1.0-beta",
)

# ── Estado global ─────────────────────────────────────────────────────────────
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
            logger.warning("Model file not found at %s — predictions disabled", path)
    return _model_bundle


@app.on_event("startup")
def startup():
    load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────

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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.get("/standings")
def standings():
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT s.position, t.name, t.shortname,
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
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT m.match_id, m.kickoff_at, m.gameweek_week,
                   ht.name AS home_team, at2.name AS away_team,
                   p.prob_home, p.prob_draw, p.prob_away
            FROM matches m
            JOIN teams ht  ON ht.team_id  = m.home_team_id
            JOIN teams at2 ON at2.team_id = m.away_team_id
            LEFT JOIN LATERAL (
                SELECT prob_home, prob_draw, prob_away
                FROM predictions
                WHERE match_id = m.match_id
                ORDER BY predicted_at DESC LIMIT 1
            ) p ON TRUE
            WHERE m.status = 'scheduled' AND m.competition_main = TRUE
            ORDER BY m.kickoff_at
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"matches": [dict(r) for r in rows]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bundle = load_model()
    if bundle is None:
        raise HTTPException(503, "Model not available")

    with get_engine().connect() as conn:
        feat_row = conn.execute(text("""
            SELECT f.*
            FROM match_features f
            JOIN matches m USING (match_id)
            WHERE (m.home_team_id = :home AND m.away_team_id = :away)
            ORDER BY m.kickoff_at DESC LIMIT 1
        """), {"home": req.home_team_id, "away": req.away_team_id}).mappings().first()

    if feat_row is None:
        feat_values = {c: 0.0 for c in bundle["feature_cols"]}
    else:
        feat_values = {c: (feat_row[c] or 0.0) for c in bundle["feature_cols"]}

    X = pd.DataFrame([feat_values])[bundle["feature_cols"]]
    probs_raw = bundle["model"].predict_proba(X)[0]
    classes   = list(bundle["model"].classes_)
    prob_map  = dict(zip(classes, probs_raw))

    return PredictResponse(
        home_team_id=req.home_team_id,
        away_team_id=req.away_team_id,
        prob_home=round(float(prob_map.get("home", 0)), 4),
        prob_draw=round(float(prob_map.get("draw", 0)), 4),
        prob_away=round(float(prob_map.get("away", 0)), 4),
        model="lgbm_v1",
    )
