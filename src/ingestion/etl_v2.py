"""ETL v2 para LaLiga Analytics.

Flujo de extraccion basado en endpoints reales verificados (HAR 2025-04-29):

  1. subscription    -> equipos maestros + gameweeks (sin partidos)
  2. standing        -> clasificacion actual
  3. teams/stats     -> stats de equipos (team_stats)
  4. players/stats   -> stats de jugadores (player_stats, paginado)
  5. /matches?subscriptionSlug={slug}&week={1..38}
                     -> partidos CON home_score/away_score incluidos
                        Paralelizado con ThreadPoolExecutor (workers=6)

Descartado definitivamente:
  x /subscriptions/{slug}/results     -> 404
  x /matches?subscriptionId=...       -> funciona pero SIN scores
  x /matches/{id}                     -> 404
  x webview /matches                  -> 404
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.clients.laliga_api import LaLigaClient
from src.storage.repository import PostgresRawRepository

logger = logging.getLogger(__name__)

MAX_WORKERS = 6   # peticiones paralelas a la API LaLiga


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
        sleep: float = 0.1,   # reducido: el paralelismo ya regula el ritmo
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
        """Thread-safe: serializa escrituras a BD con Lock."""
        with self._db_lock:
            self.repo.save(
                resource=resource,
                payload=payload,
                competition_slug=self.season.competition_slug,
                season_label=self.season.season_label,
            )

    def _make_client(self) -> LaLigaClient:
        """Crea un cliente HTTP independiente para cada thread."""
        return LaLigaClient(
            subscription_key=self.client.subscription_key,
            base_url=self.client.base_url,
        )

    # ------------------------------------------------------------------
    # Pasos secuenciales (no paralelizables entre si)
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
        """Ejecutado en un thread: descarga una jornada y devuelve (week, payload)."""
        client = self._make_client()
        payload = client.get_matches_by_week(self.season.subscription_slug, week)
        time.sleep(self.sleep)   # cortesia minima por thread
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
    run_season(
        competition_slug="primera-division",
        season_label="2025",
        subscription_slug="laliga-easports-2025",
        subscription_key="c13c3a8e2f6b46da9c5c425cf61fab3e",
    )
