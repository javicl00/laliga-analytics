"""ETL: ingesta de stats por partido desde la API publica de LaLiga (Opta webview).

Endpoint descubierto via HAR:
  GET https://apim.laliga.com/webview/api/web/matches/opta/{opta_id}/stats
      ?contentLanguage=es&countryCode=ES&subscription-key=ee7fcd5c543f4485ba2a48856fc7ece9

El campo opta_id en matches tiene formato 'g2572229'.
El endpoint construye la URL como /opta/g2572229/stats.

Ejecucion:
  docker compose run --rm etl python -m src.ingestion.fetch_match_stats
  docker compose run --rm etl python -m src.ingestion.fetch_match_stats --season 375
  docker compose run --rm etl python -m src.ingestion.fetch_match_stats --match-id 98754

Se salta automaticamente los partidos ya ingestados (idempotente).
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import create_engine, text
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BASE_URL      = "https://apim.laliga.com/webview/api/web/matches/opta"
_SUB_KEY       = "ee7fcd5c543f4485ba2a48856fc7ece9"
_RATE_LIMIT_S  = 0.2   # 200 ms entre requests (~5 req/s)
_RETRY_TOTAL   = 3
_RETRY_BACKOFF = 1.0


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=_RETRY_TOTAL,
        backoff_factor=_RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "Origin":     "https://www.laliga.com",
        "Referer":    "https://www.laliga.com/",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept":     "application/json",
    })
    return session


def fetch_stats(opta_id: str, session: requests.Session) -> Optional[List[Dict]]:
    """Llama al endpoint y devuelve lista de dicts con stats por equipo, o None si falla."""
    url = f"{_BASE_URL}/{opta_id}/stats"
    params = {
        "contentLanguage": "es",
        "countryCode":     "ES",
        "subscription-key": _SUB_KEY,
    }
    try:
        resp = session.get(url, params=params, timeout=10)
        if resp.status_code == 404:
            logger.debug("404 para %s (partido sin stats en Opta)", opta_id)
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get("match_team_stats", [])
    except requests.RequestException as exc:
        logger.warning("Error fetching %s: %s", opta_id, exc)
        return None


def _resolve_is_home(
    opta_team_id: str,
    home_opta: Optional[str],
    away_opta: Optional[str],
) -> Optional[bool]:
    """Devuelve True si opta_team_id es el equipo local, False si visitante, None si no resuelve."""
    if home_opta and opta_team_id == home_opta:
        return True
    if away_opta and opta_team_id == away_opta:
        return False
    return None


def _upsert_stat(conn, match_id: int, is_home: bool, opta_team_id: str, stats: Dict) -> None:
    s = stats
    conn.execute(text("""
        INSERT INTO match_stats_opta (
            match_id, is_home, opta_team_id,
            possession_pct, ppda,
            shots_on_target, total_shots,
            big_chances_created, big_chances_missed,
            accurate_pass, total_pass,
            aerial_won, aerial_lost
        ) VALUES (
            :match_id, :is_home, :opta_team_id,
            :possession_pct, :ppda,
            :shots_on_target, :total_shots,
            :big_chances_created, :big_chances_missed,
            :accurate_pass, :total_pass,
            :aerial_won, :aerial_lost
        )
        ON CONFLICT (match_id, is_home) DO UPDATE SET
            opta_team_id      = EXCLUDED.opta_team_id,
            possession_pct    = EXCLUDED.possession_pct,
            ppda              = EXCLUDED.ppda,
            shots_on_target   = EXCLUDED.shots_on_target,
            total_shots       = EXCLUDED.total_shots,
            big_chances_created = EXCLUDED.big_chances_created,
            big_chances_missed  = EXCLUDED.big_chances_missed,
            accurate_pass     = EXCLUDED.accurate_pass,
            total_pass        = EXCLUDED.total_pass,
            aerial_won        = EXCLUDED.aerial_won,
            aerial_lost       = EXCLUDED.aerial_lost,
            fetched_at        = now()
    """), {
        "match_id":   match_id,
        "is_home":    is_home,
        "opta_team_id": opta_team_id,
        "possession_pct":      s.get("possession_percentage"),
        "ppda":                s.get("ppda"),
        "shots_on_target":     s.get("ontargetscoringatt") or s.get("shots_on_target"),
        "total_shots":         s.get("totalscoringatt") or s.get("total_scoring_att"),
        "big_chances_created": s.get("bigchancecreated") or s.get("big_chance_created"),
        "big_chances_missed":  s.get("bigchancemissed") or s.get("big_chance_missed"),
        "accurate_pass":       s.get("accuratepass") or s.get("accurate_pass"),
        "total_pass":          s.get("totalpass") or s.get("total_pass"),
        "aerial_won":          s.get("aerialwon") or s.get("aerial_won"),
        "aerial_lost":         s.get("aeriallost") or s.get("aerial_lost"),
    })


def run(
    engine,
    season_id:  Optional[int] = None,
    match_id:   Optional[int] = None,
    force:      bool = False,
) -> Dict:
    """Ingesta stats para los partidos que coincidan con los filtros.

    Args:
        engine:    SQLAlchemy engine.
        season_id: si se especifica, solo partidos de esa temporada.
        match_id:  si se especifica, solo ese partido.
        force:     si True, re-ingesta aunque ya existan datos.
    """
    # Construir query de partidos a procesar
    where_clauses = [
        "m.opta_id IS NOT NULL",
        "m.competition_main = TRUE",
        "m.status IN ('FullTime', 'FinishedPeriod', 'Finished')",
    ]
    params: Dict = {}

    if season_id:
        where_clauses.append("m.season_id = :season_id")
        params["season_id"] = season_id
    if match_id:
        where_clauses.append("m.match_id = :match_id")
        params["match_id"] = match_id
    if not force:
        # Solo partidos sin las dos filas home+away en match_stats_opta
        where_clauses.append("""
            (SELECT COUNT(*) FROM match_stats_opta mso WHERE mso.match_id = m.match_id) < 2
        """)

    where_sql = " AND ".join(where_clauses)
    query = text(f"""
        SELECT
            m.match_id,
            m.opta_id,
            ht.opta_id AS home_opta_team_id,
            at.opta_id AS away_opta_team_id
        FROM matches m
        LEFT JOIN teams ht ON ht.team_id = m.home_team_id
        LEFT JOIN teams at ON at.team_id = m.away_team_id
        WHERE {where_sql}
        ORDER BY m.kickoff_at
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    logger.info("Partidos a procesar: %d", len(rows))
    if not rows:
        return {"processed": 0, "ok": 0, "not_found": 0, "error": 0}

    session = _build_session()
    ok = not_found = error = 0

    for row in rows:
        mid        = row.match_id
        opta_id    = row.opta_id          # 'g2572229'
        home_opta  = row.home_opta_team_id  # 't184'
        away_opta  = row.away_opta_team_id  # 't954'

        team_stats = fetch_stats(opta_id, session)
        time.sleep(_RATE_LIMIT_S)

        if team_stats is None:
            not_found += 1
            continue

        if len(team_stats) != 2:
            logger.warning("match %d: %d equipos en stats (esperado 2)", mid, len(team_stats))
            error += 1
            continue

        try:
            with engine.begin() as conn:
                for entry in team_stats:
                    ot_id   = entry.get("opta_team_id", "")
                    stats   = entry.get("stats", {})
                    is_home = _resolve_is_home(ot_id, home_opta, away_opta)

                    if is_home is None:
                        # Fallback: asignar por orden (primer entry = home)
                        is_home = (team_stats.index(entry) == 0)
                        logger.debug(
                            "match %d: no se pudo resolver is_home para %s, fallback=%s",
                            mid, ot_id, is_home,
                        )

                    _upsert_stat(conn, mid, is_home, ot_id, stats)

            logger.debug("match %d (%s): OK", mid, opta_id)
            ok += 1
        except Exception as exc:
            logger.error("match %d: error al persistir: %s", mid, exc)
            error += 1

    result = {"processed": len(rows), "ok": ok, "not_found": not_found, "error": error}
    logger.info("Resultado: %s", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest match stats from LaLiga Opta API")
    parser.add_argument("--season",   type=int, help="Filtrar por season_id")
    parser.add_argument("--match-id", type=int, help="Procesar solo este match_id")
    parser.add_argument("--force",    action="store_true", help="Re-ingestar aunque ya existan datos")
    args = parser.parse_args()

    eng = create_engine(os.environ["DATABASE_URL"])
    run(eng, season_id=args.season, match_id=args.match_id, force=args.force)
