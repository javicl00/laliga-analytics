"""API de serving para predicciones LaLiga Analytics."""
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

app = FastAPI(title="LaLiga Analytics", version="1.0.0")

class PredictRequest(BaseModel):
    season: int
    home_team_id: int
    away_team_id: int
    kickoff_at: datetime

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/match")
def predict_match(req: PredictRequest):
    return {
        "message": "endpoint ready - connect model predictor",
        "request": req.model_dump(),
    }
