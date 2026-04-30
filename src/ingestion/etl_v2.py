"""ETL v2 para LaLiga Analytics.

Flujo de extraccion basado en endpoints reales verificados (HAR 2025-04-30):

  1. subscription    -> equipos maestros + gameweeks (sin partidos)
  2. standing        -> clasificacion actual
  3. teams/stats     -> stats de equipos (team_stats)
  4. players/stats   -> stats de jugadores (player_stats, paginado)
  5. /matches?subscriptionSlug={slug}&week={1..38}
                     -> partidos CON home_score/away_score incluidos
                        Paralelizado con ThreadPoolExecutor (workers=6)

Slugs verificados canonicamente via GET /subscriptions (2026-04-30):
  id=1    laliga-santander-2013   2013-2014
  id=10   laliga-santander-2014   2014-2015
  id=19   laliga-santander-2015   2015-2016
  id=30   laliga-santander-2016   2016-2017
  id=41   laliga-santander-2017   2017-2018
  id=56   laliga-santander-2018   2018-2019
  id=82   laliga-santander-2019   2019-2020
  id=97   laliga-santander-2020   2020-2021
  id=116  laliga-santander-2021   2021-2022
  id=305  laliga-santander-2022   2022-2023
  id=329  laliga-easports-2023    2023-2024
  id=351  laliga-easports-2024    2024-2025
  id=375  laliga-easports-2025    2025-2026  (en curso)

Descartado definitivamente:
  x /subscriptions/{slug}/results     -> 404
  x /matches?subscriptionId=...       -> funciona pero SIN scores
  x /matches/{id}                     -> 404
  x laliga-easports-2022              -> no existe (era LaLiga Santander ese año)
  x Datos anteriores a 2013           -> no disponibles en la API
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.clients.laliga_api import LaLigaClient
from src.storage.repository import PostgresRawRepository

logger = logging.getLogger(__name__)

MAX_WORKERS = 6

# Mapa canónico completo verificado via GET /subscriptions (competition.id=1)
KNOWN_SEASONS = [
    {"label": "2013", "slug": "laliga-santander-2013", "sub_id": 1},
    {"label": "2014", "slug": "laliga-santander-2014", "sub_id": 10},
    {"label": "2015", "slug": "laliga-santander-2015", "sub_id": 19},
    {"label": "2016", "slug": "laliga-santander-2016", "sub_id": 30},
    {"label": "2017", "slug": "laliga-santander-2017", "sub_id": 41},
    {"label": "2018", "slug": "laliga-santander-2018", "sub_id": 56},
    {"label": "2019", "slug": "laliga-santander-2019", "sub_id": 82},
    {"label": "2020", "slug": "laliga-santander-2020", "sub_id": 97},
    {"label": "2021", "slug": "laliga-santander-2021", "sub_id": 116},
    {"label": "2022", "slug": "laliga-santander-2022", "sub_id": 305},
    {"label": "2023", "slug": "laliga-easports-2023",  "sub_id": 329},
    {"label": "2024", "slug": "laliga-easports-2024",  "sub_id": 351},
    {"label": "2025", "slug": "laliga-easports-2025",  "sub_id": 375},
]


@dataclass
class SeasonConfig:
    competition_slug: str
    season_label: str
    subscription_slug: str
    num_gameweeks: int = 38
    main_only: bool = True
    fetch_match_stats: bool = False
    fetch_match_events: bool = False
    weeks: List[int] = field(default_factory=list)


class ETLv2:
    """ETL completo basado en endpoints verificados.

    El metodo _extract_matches_by_week usa ThreadPoolExecutor para
    lanzar hasta MAX_WORKERS peticiones en paralelo. Cada thread
    tiene su propio LaLigaClient (sesion HTTP independiente).
    El acceso a PostgresRawRepository esta serializado con un Lock.
    """

    def __init__(
        self,
        client: LaLigaClient,
        repo: PostgresRawRepository,
        season: SeasonConfig,
        sleep: float = 0.1,
    ) -> None:
        self.client = client
        self.repo = repo
        self.season = season
        self.sleep = sleep
        self._db_lock = threading.Lock()
        self._snapshot_ts = datetime.now(timezone.utc).isoformat()

    def run(self) -> None:
        logger.info("ETL v2 start | %s %s", self.season.competition_slug, self.season.season_label)
        self._extract_subscription()
        self._extract_standing()
        self._extract_teams_stats()
        self._extract_players_stats()
        weeks = self.season.weeks or list(range(1, self.season.num_gameweeks + 1))
        self._extract_matches_by_week(weeks)
        logger.info("ETL v2 complete | %s %s", self.season.competition_slug, self.season.season_label)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, resource: str, payload) -> None:
        with self._db_lock:
            self.repo.save(
                resource=resource,
                payload=payload,
                competition_slug=self.season.competition_slug,
                season_label=self.season.season_label,
            )

    def _make_client(self) -> LaLigaClient:
        return LaLigaClient(
            subscription_key=self.client.subscription_key,
            base_url=self.client.base_url,
        )

    # ------------------------------------------------------------------
    # Pasos secuenciales
    # ------------------------------------------------------------------

    def _extract_subscription(self):
        logger.info("Extracting subscription")
        payload = self.client.get_subscription(self.season.subscription_slug)
        self._save("subscription", payload)

    def _extract_standing(self):
        logger.info("Extracting standing")
        self._save("standing", self.client.get_standing(self.season.subscription_slug))

    def _extract_teams_stats(self):
        logger.info("Extracting teams/stats")
        teams = self.client.get_all_teams_stats(self.season.subscription_slug)
        self._save("teams_stats", {"team_stats": teams})

    def _extract_players_stats(self):
        logger.info("Extracting players/stats (paginated)")
        players = self.client.get_all_players_stats(self.season.subscription_slug)
        self._save("players_stats", {"player_stats": players})

    # ------------------------------------------------------------------
    # Extraccion paralela de jornadas
    # ------------------------------------------------------------------

    def _fetch_week(self, week: int) -> tuple[int, dict]:
        client = self._make_client()
        payload = client.get_matches_by_week(self.season.subscription_slug, week)
        time.sleep(self.sleep)
        return week, payload

    def _extract_matches_by_week(self, weeks: List[int]) -> None:
        total = len(weeks)
        logger.info("Extracting %d weeks with %d parallel workers", total, MAX_WORKERS)
        t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(self._fetch_week, w): w for w in weeks}
            done = 0
            for future in as_completed(futures):
                week = futures[future]
                try:
                    week, payload = future.result()
                    self._save(f"matches_week_{week}", payload)
                    done += 1
                    logger.info("[%d/%d] week=%d matches=%d",
                                done, total, week,
                                len(payload.get("matches", [])))
                except Exception as exc:
                    logger.error("week=%d failed: %s", week, exc)

        elapsed = time.monotonic() - t0
        logger.info("Matches extraction done in %.1fs", elapsed)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def run_season(
    competition_slug: str,
    season_label: str,
    subscription_slug: str,
    subscription_key: str,
    num_gameweeks: int = 38,
    db_url: Optional[str] = None,
) -> None:
    client = LaLigaClient(subscription_key=subscription_key)
    repo = PostgresRawRepository(db_url=db_url)
    season = SeasonConfig(
        competition_slug=competition_slug,
        season_label=season_label,
        subscription_slug=subscription_slug,
        num_gameweeks=num_gameweeks,
    )
    ETLv2(client=client, repo=repo, season=season).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="LaLiga ETL v2")
    parser.add_argument(
        "--season",
        choices=[s["label"] for s in KNOWN_SEASONS] + ["all"],
        default="all",
        help="Temporada a extraer (2013-2025) o 'all' para todas (default: all)",
    )
    args = parser.parse_args()

    subscription_key = os.environ.get("LALIGA_API_KEY", "c13c3a8e2f6b46da9c5c425cf61fab3e")

    seasons_to_run = (
        KNOWN_SEASONS
        if args.season == "all"
        else [s for s in KNOWN_SEASONS if s["label"] == args.season]
    )

    for s in seasons_to_run:
        run_season(
            competition_slug="primera-division",
            season_label=s["label"],
            subscription_slug=s["slug"],
            subscription_key=subscription_key,
        )
