"""ETL v2 para LaLiga Analytics.

Flujo de extracción basado en endpoints reales verificados (HAR 2025-04-29):

  1. subscription    → equipos maestros + gameweeks (sin partidos)
  2. standing        → clasificación actual
  3. teams/stats     → stats de equipos (team_stats)
  4. players/stats   → stats de jugadores (player_stats, paginado)
  5. /matches?subscriptionSlug={slug}&week={1..38}
                     → partidos CON home_score/away_score incluidos
                       NO requiere fan-out a /matches/{id}

Descartado definitivamente:
  ✘ /subscriptions/{slug}/results     → 404
  ✘ /matches?subscriptionId=...       → funciona pero SIN scores
  ✘ /matches/{id}                     → 404
  ✘ webview /matches                  → 404
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.clients.laliga_api import LaLigaClient
from src.storage.repository import RawRepository

logger = logging.getLogger(__name__)


@dataclass
class SeasonConfig:
    competition_slug: str
    season_label: str
    subscription_slug: str
    num_gameweeks: int = 38          # Jornadas regulares de liga
    main_only: bool = True           # Filtrar solo competition.main=True en /matches
    fetch_match_stats: bool = False  # GET /matches/{id}/stats (puede ser 404)
    fetch_match_events: bool = False # GET /matches/{id}/events (puede ser 404)
    weeks: List[int] = field(default_factory=list)  # Si vacio, usa range(1, num_gameweeks+1)


class ETLv2:
    """ETL completo basado en endpoints verificados.

    Parameters
    ----------
    client: LaLigaClient
    repo:   RawRepository
    season: SeasonConfig
    sleep:  Segundos entre requests (respetar rate-limit API)
    """

    def __init__(
        self,
        client: LaLigaClient,
        repo: RawRepository,
        season: SeasonConfig,
        sleep: float = 0.4,
    ) -> None:
        self.client = client
        self.repo = repo
        self.season = season
        self.sleep = sleep
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
    # Pasos
    # ------------------------------------------------------------------

    def _save(self, resource: str, payload) -> None:
        self.repo.save(
            resource=resource,
            payload=payload,
            competition_slug=self.season.competition_slug,
            season_label=self.season.season_label,
        )
        time.sleep(self.sleep)

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

    def _extract_matches_by_week(self, weeks: List[int]) -> None:
        """Extrae partidos jornada a jornada.

        Cada payload incluye home_score/away_score directamente.
        NO se necesita fan-out a /matches/{id}.
        """
        total = len(weeks)
        for i, week in enumerate(weeks, 1):
            logger.info("Extracting matches week %d/%d (week=%s)", i, total, week)
            payload = self.client.get_matches_by_week(
                self.season.subscription_slug, week
            )
            self._save(f"matches_week_{week}", payload)

            # Fan-out opcional a stats/events por partido (probablemente 404)
            if self.season.fetch_match_stats or self.season.fetch_match_events:
                self._fan_out_optional(payload)

    def _fan_out_optional(self, matches_payload: dict) -> None:
        """Fan-out opcional a /matches/{id}/stats y /events."""
        for m in matches_payload.get("matches", []):
            if m.get("status") not in ("FinishedPeriod", "FullTime", "Finished"):
                continue
            match_id = m.get("id")
            if not match_id:
                continue
            if self.season.fetch_match_stats:
                stats = self.client.get_match_stats(match_id)
                if stats:
                    self._save(f"match_stats_{match_id}", stats)
            if self.season.fetch_match_events:
                events = self.client.get_match_events(match_id)
                if events:
                    self._save(f"match_events_{match_id}", events)


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
    from src.storage.repository import PostgresRawRepository
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
