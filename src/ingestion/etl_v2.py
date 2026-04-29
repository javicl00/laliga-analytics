"""ETL v2 para LaLiga Analytics.

Basa la extracción en los endpoints reales verificados a 2025-04-29:

  1. subscription    → equipos maestros + gameweeks (sin partidos)
  2. standing        → clasificación actual
  3. teams/stats     → stats de equipos (clave: team_stats)
  4. players/stats   → stats de jugadores (clave: player_stats, paginado)
  5. /matches?subscriptionId={slug}&gameweekId={gw_id}
                     → partidos por jornada (20 items, incluye otras competiciones)
  6. /matches/{id}   → detalle con score (fan-out por partido terminado)

NOTA ARQUITECTÓNICA:
  - /subscriptions/{slug}/results → 404. No usar.
  - Los goles NO vienen en el endpoint de partidos por jornada.
    Se necesita fan-out a /matches/{id} para cada partido FinishedPeriod.
  - competition_id de LaLiga se infiere desde global_data o subscription.
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
    laliga_competition_id: Optional[int] = None  # Filtro para /matches
    fetch_match_detail: bool = True              # Fan-out a /matches/{id} para scores
    fetch_match_stats: bool = False
    fetch_match_events: bool = False
    gameweek_ids: List[int] = field(default_factory=list)


class ETLv2:
    """ETL completo basado en endpoints verificados.

    Parameters
    ----------
    client: LaLigaClient
    repo:   RawRepository
    season: SeasonConfig
    sleep:  Segundos entre requests
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

        sub_payload = self._extract_subscription()
        self._extract_standing()
        self._extract_teams_stats()
        self._extract_players_stats()

        # Extraer gameweek_ids si no se pasaron en la config
        gw_ids = self.season.gameweek_ids
        if not gw_ids and sub_payload:
            gw_ids = self._extract_gameweek_ids(sub_payload)
            logger.info("Discovered %d gameweeks from subscription", len(gw_ids))

        if gw_ids:
            self._extract_matches_by_gameweek(gw_ids)

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
        return payload

    def _extract_standing(self):
        logger.info("Extracting standing")
        payload = self.client.get_standing(self.season.subscription_slug)
        self._save("standing", payload)

    def _extract_teams_stats(self):
        logger.info("Extracting teams/stats")
        teams = self.client.get_all_teams_stats(self.season.subscription_slug)
        self._save("teams_stats", {"team_stats": teams})

    def _extract_players_stats(self):
        logger.info("Extracting players/stats (paginated)")
        players = self.client.get_all_players_stats(self.season.subscription_slug)
        self._save("players_stats", {"player_stats": players})

    def _extract_gameweek_ids(self, sub_payload: dict) -> List[int]:
        """Extrae todos los gameweek_id desde subscription.rounds[].gameweeks[]."""
        sub = sub_payload.get("subscription", sub_payload)
        gw_ids = []
        seen = set()
        for r in sub.get("rounds", []):
            for gw in r.get("gameweeks", []):
                gw_id = gw.get("id")
                if gw_id and gw_id not in seen:
                    seen.add(gw_id)
                    gw_ids.append(gw_id)
        return sorted(gw_ids)

    def _extract_matches_by_gameweek(self, gw_ids: List[int]) -> None:
        """Extrae partidos jornada a jornada y hace fan-out a detalle si procede."""
        total = len(gw_ids)
        all_match_ids_finished: List[int] = []

        for i, gw_id in enumerate(gw_ids, 1):
            logger.info("Extracting matches gw %d/%d (id=%s)", i, total, gw_id)
            payload = self.client.get_matches_by_gameweek(
                self.season.subscription_slug, gw_id
            )
            self._save(f"matches_gw_{gw_id}", payload)

            # Recopilar match_ids de partidos terminados para fan-out
            if self.season.fetch_match_detail:
                for m in payload.get("matches", []):
                    # Filtro por competition si se especificó
                    if self.season.laliga_competition_id is not None:
                        comp_id = (m.get("competition") or {}).get("id")
                        if comp_id != self.season.laliga_competition_id:
                            continue
                    if m.get("status") in ("FinishedPeriod", "FullTime", "Finished"):
                        all_match_ids_finished.append(m["id"])

        if all_match_ids_finished:
            self._fan_out_match_detail(all_match_ids_finished)

    def _fan_out_match_detail(self, match_ids: List[int]) -> None:
        """Fan-out a GET /matches/{id} para obtener scores."""
        total = len(match_ids)
        logger.info("Fan-out match detail for %d finished matches", total)
        for i, match_id in enumerate(match_ids, 1):
            logger.debug("Match detail %d/%d id=%s", i, total, match_id)
            payload = self.client.get_match_detail(match_id)
            if payload:
                self._save(f"match_detail_{match_id}", payload)

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
    laliga_competition_id: Optional[int] = None,
    fetch_match_detail: bool = True,
    db_url: Optional[str] = None,
) -> None:
    from src.storage.repository import PostgresRawRepository
    client = LaLigaClient(subscription_key=subscription_key)
    repo = PostgresRawRepository(db_url=db_url)
    season = SeasonConfig(
        competition_slug=competition_slug,
        season_label=season_label,
        subscription_slug=subscription_slug,
        laliga_competition_id=laliga_competition_id,
        fetch_match_detail=fetch_match_detail,
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
