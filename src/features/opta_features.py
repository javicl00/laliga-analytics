"""Familia G: features rolling de stats Opta por partido (prepartido, sin leakage).

Requiere que match_stats_opta este poblada por fetch_match_stats.py.

Algoritmo vectorizado (O(n log n)):
  1. Construye tabla long: (match_id, team_id, kickoff_at, is_home, stats...)
  2. Ordena por team_id + kickoff_at
  3. groupby(team_id).shift(1).rolling(5).mean()  → rolling sin leakage
  4. Pivot de vuelta a columnas home_ / away_ por match_id

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
    "possession_pct":    "possession_last5",
    "ppda":              "ppda_last5",
    "shots_on_target":   "shots_ot_last5",
    "big_chances_created": "bigchances_last5",
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
    """Construye tabla long: una fila por (partido, equipo) con sus stats."""
    # Stats home
    home_stats = stats[stats["is_home"]].merge(
        matches[["match_id", "kickoff_at", "home_team_id"]],
        on="match_id",
    ).rename(columns={"home_team_id": "team_id"}).drop(columns="is_home")

    # Stats away
    away_stats = stats[~stats["is_home"]].merge(
        matches[["match_id", "kickoff_at", "away_team_id"]],
        on="match_id",
    ).rename(columns={"away_team_id": "team_id"}).drop(columns="is_home")

    long = pd.concat([home_stats, away_stats], ignore_index=True)
    return long.sort_values(["team_id", "kickoff_at"]).reset_index(drop=True)


def compute_opta_features(engine) -> pd.DataFrame:
    """Devuelve DataFrame con match_id + 8 columnas familia G.

    Usa shift(1) antes del rolling para garantizar que el partido actual
    no entra en su propio calculo (sin leakage).
    """
    matches, stats = _load_data(engine)

    if stats.empty:
        logger.warning("match_stats_opta vacia, no hay features que calcular")
        return pd.DataFrame()

    long = _build_team_long(matches, stats)

    # Rolling avg ultimos N partidos ANTERIORES (shift(1) elimina el partido actual)
    rolled = (
        long
        .groupby("team_id", sort=False)[_STAT_COLS]
        .transform(lambda x: x.shift(1).rolling(ROLLING_N, min_periods=1).mean())
    )
    long[[f"{c}_r" for c in _STAT_COLS]] = rolled

    rolled_cols = [f"{c}_r" for c in _STAT_COLS]

    # Join rolling stats al DataFrame de partidos para home y away
    home_rolled = long[["match_id"] + rolled_cols].merge(
        matches[["match_id", "home_team_id"]],
        on="match_id",
    )
    # Puede haber duplicados si un equipo aparece varias veces; nos quedamos
    # con el registro home del partido correcto
    home_stats_full = stats[stats["is_home"]].merge(
        matches[["match_id", "home_team_id", "kickoff_at"]],
        on="match_id",
    ).rename(columns={"home_team_id": "team_id"})

    away_stats_full = stats[~stats["is_home"]].merge(
        matches[["match_id", "away_team_id", "kickoff_at"]],
        on="match_id",
    ).rename(columns={"away_team_id": "team_id"})

    long2 = pd.concat(
        [home_stats_full.assign(role="home"), away_stats_full.assign(role="away")],
        ignore_index=True,
    ).sort_values(["team_id", "kickoff_at"]).reset_index(drop=True)

    rolled2 = (
        long2
        .groupby("team_id", sort=False)[_STAT_COLS]
        .transform(lambda x: x.shift(1).rolling(ROLLING_N, min_periods=1).mean())
    )
    long2[[f"{c}_r" for c in _STAT_COLS]] = rolled2

    home_part = long2[long2["role"] == "home"][["match_id"] + rolled_cols].copy()
    home_part.columns = ["match_id"] + [f"home_{_FEAT_MAP[c.rstrip('_r').replace('_r','')]}"
                                         if c != "match_id" else c
                                         for c in rolled_cols]

    away_part = long2[long2["role"] == "away"][["match_id"] + rolled_cols].copy()
    away_part.columns = ["match_id"] + [f"away_{_FEAT_MAP[c.rstrip('_r').replace('_r','')]}"
                                         if c != "match_id" else c
                                         for c in rolled_cols]

    # Renombrar correctamente
    home_rename = {f"{c}_r": f"home_{_FEAT_MAP[c]}" for c in _STAT_COLS}
    away_rename = {f"{c}_r": f"away_{_FEAT_MAP[c]}" for c in _STAT_COLS}

    home_part = long2[long2["role"] == "home"][["match_id"] + rolled_cols].rename(columns=home_rename)
    away_part = long2[long2["role"] == "away"][["match_id"] + rolled_cols].rename(columns=away_rename)

    result = matches[["match_id"]].merge(home_part, on="match_id", how="left") \
                                  .merge(away_part,  on="match_id", how="left")

    # Filtrar partidos donde todo es NaN (sin historial Opta)
    feat_cols = list(home_rename.values()) + list(away_rename.values())
    result = result.dropna(subset=feat_cols, how="all")

    logger.info("Opta features calculadas para %d partidos", len(result))
    return result


def write_to_db(df: pd.DataFrame, engine) -> None:
    """Escribe las features en match_features via UPDATE con executemany."""
    if df.empty:
        logger.warning("DataFrame vacio, nada que escribir")
        return

    feat_cols = [
        "home_possession_last5", "away_possession_last5",
        "home_ppda_last5",       "away_ppda_last5",
        "home_shots_ot_last5",   "away_shots_ot_last5",
        "home_bigchances_last5", "away_bigchances_last5",
    ]
    # Asegurar que existen todas las columnas (pueden faltar si no hay stats)
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
        conn.execute(update_sql, records)   # executemany automatico

    logger.info("match_features actualizado para %d partidos", len(df))


if __name__ == "__main__":
    eng = create_engine(os.environ["DATABASE_URL"])
    df  = compute_opta_features(eng)
    write_to_db(df, eng)
    print(df.head())
