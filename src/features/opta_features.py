"""Familia G: features rolling de stats Opta por partido (prepartido, sin leakage).

Requiere que match_stats_opta este poblada por fetch_match_stats.py.

Algoritmo vectorizado (O(n log n)):
  1. Construye tabla long desde TODOS los matches (home + away) con LEFT JOIN a stats
  2. Ordena por team_id + kickoff_at
  3. groupby(team_id).shift(1).rolling(5).mean() -> rolling sin leakage
     shift(1) garantiza que el partido actual no entra en su propio calculo
  4. Pivot de vuelta a columnas home_ / away_ por match_id

Nota: un partido sin stats propias PUEDE tener rolling features si el equipo
tiene historial previo con stats -> se mantienen todos los partidos posibles.

Features generadas (8 columnas):
  home/away_possession_last5  -- % posesion media ultimos 5
  home/away_ppda_last5        -- PPDA medio (menor = mas presion)
  home/away_shots_ot_last5    -- tiros a puerta medios
  home/away_bigchances_last5  -- grandes ocasiones creadas medias

Ejecucion:
  docker compose run --rm etl python -m src.features.opta_features
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLLING_N  = 5
_STAT_COLS = ["possession_pct", "ppda", "shots_on_target", "big_chances_created"]
_FEAT_MAP  = {
    "possession_pct":       "possession_last5",
    "ppda":                 "ppda_last5",
    "shots_on_target":      "shots_ot_last5",
    "big_chances_created":  "bigchances_last5",
}


def _load_data(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    with engine.connect() as conn:
        matches = pd.read_sql(text(
            "SELECT match_id, kickoff_at, home_team_id, away_team_id "
            "FROM matches WHERE competition_main = TRUE ORDER BY kickoff_at"
        ), conn)
        stats = pd.read_sql(text(
            "SELECT match_id, is_home, possession_pct, ppda, "
            "shots_on_target, big_chances_created "
            "FROM match_stats_opta"
        ), conn)
    matches["kickoff_at"] = pd.to_datetime(matches["kickoff_at"], utc=True)
    return matches, stats


def _build_team_long(matches: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Tabla long con UNA fila por (partido, rol) para todos los matches.

    LEFT JOIN a stats: partidos sin stats propias tienen NaN en stat_cols
    pero siguen presentes para que el rolling funcione correctamente.
    """
    # Fila home para cada partido
    home = matches[["match_id", "kickoff_at", "home_team_id"]].rename(
        columns={"home_team_id": "team_id"}
    ).assign(role="home")

    # Fila away para cada partido
    away = matches[["match_id", "kickoff_at", "away_team_id"]].rename(
        columns={"away_team_id": "team_id"}
    ).assign(role="away")

    long = pd.concat([home, away], ignore_index=True)

    # LEFT JOIN a stats
    stats_home = stats[stats["is_home"] == True].drop(columns="is_home")   # noqa: E712
    stats_away = stats[stats["is_home"] == False].drop(columns="is_home")  # noqa: E712

    long_home = long[long["role"] == "home"].merge(
        stats_home, on="match_id", how="left"
    )
    long_away = long[long["role"] == "away"].merge(
        stats_away, on="match_id", how="left"
    )

    result = pd.concat([long_home, long_away], ignore_index=True)
    return result.sort_values(["team_id", "kickoff_at"]).reset_index(drop=True)


def compute_opta_features(engine) -> pd.DataFrame:
    """Devuelve DataFrame con match_id + 8 columnas familia G."""
    matches, stats = _load_data(engine)

    if stats.empty:
        logger.warning("match_stats_opta vacia, no hay features que calcular")
        return pd.DataFrame()

    long = _build_team_long(matches, stats)

    # Rolling avg: shift(1) excluye el partido actual, min_periods=1 evita NaN
    # cuando hay menos de ROLLING_N partidos previos con stats
    rolled = (
        long
        .groupby("team_id", sort=False)[_STAT_COLS]
        .transform(lambda x: x.shift(1).rolling(ROLLING_N, min_periods=1).mean())
    )
    rolled_cols = {c: f"{c}_r" for c in _STAT_COLS}
    long = long.join(rolled.rename(columns=rolled_cols))

    # Separar home y away con sus rolling features
    home_rename = {f"{c}_r": f"home_{_FEAT_MAP[c]}" for c in _STAT_COLS}
    away_rename = {f"{c}_r": f"away_{_FEAT_MAP[c]}" for c in _STAT_COLS}

    rcols = list(rolled_cols.values())  # ['possession_pct_r', ...]

    home_part = (
        long[long["role"] == "home"][["match_id"] + rcols]
        .rename(columns=home_rename)
    )
    away_part = (
        long[long["role"] == "away"][["match_id"] + rcols]
        .rename(columns=away_rename)
    )

    result = (
        matches[["match_id"]]
        .merge(home_part, on="match_id", how="left")
        .merge(away_part,  on="match_id", how="left")
    )

    # Descartar partidos donde TODOS los valores son NaN (sin historial Opta)
    feat_cols = list(home_rename.values()) + list(away_rename.values())
    result = result.dropna(subset=feat_cols, how="all")

    logger.info("Opta features calculadas para %d partidos", len(result))
    return result


def write_to_db(df: pd.DataFrame, engine) -> None:
    """Escribe las features en match_features via UPDATE (executemany)."""
    if df.empty:
        logger.warning("DataFrame vacio, nada que escribir")
        return

    feat_cols = [
        "home_possession_last5", "away_possession_last5",
        "home_ppda_last5",       "away_ppda_last5",
        "home_shots_ot_last5",   "away_shots_ot_last5",
        "home_bigchances_last5", "away_bigchances_last5",
    ]
    for col in feat_cols:
        if col not in df.columns:
            df[col] = None

    update_sql = text("""
        UPDATE match_features SET
            home_possession_last5  = :home_possession_last5,
            away_possession_last5  = :away_possession_last5,
            home_ppda_last5        = :home_ppda_last5,
            away_ppda_last5        = :away_ppda_last5,
            home_shots_ot_last5    = :home_shots_ot_last5,
            away_shots_ot_last5    = :away_shots_ot_last5,
            home_bigchances_last5  = :home_bigchances_last5,
            away_bigchances_last5  = :away_bigchances_last5
        WHERE match_id = :match_id
    """)

    records = df[["match_id"] + feat_cols].to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(update_sql, records)

    logger.info("match_features actualizado para %d partidos", len(df))


if __name__ == "__main__":
    eng = create_engine(os.environ["DATABASE_URL"])
    df  = compute_opta_features(eng)
    write_to_db(df, eng)
    print(df.head())
