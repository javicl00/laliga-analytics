"""ETL v1 para LaLiga Analytics.

Extrae resultados, clasificación, estadísticas de equipo y jugador
para una temporada concreta. Deja preparado el fan-out para match stats
y match events cuando estén disponibles en la API.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.clients.laliga_api import LaLigaClient
from src.storage.repository import RawRepository

logger = logging.getLogger(__name__)


@dataclass
class SeasonConfig:
    """Configuración de una temporada a extraer."""
    competition_slug: str
    season_label: str          # e.g. "2025"
    subscription_slug: str    # e.g. "laliga-easports-2025"
    has_match_stats: bool = False
    has_match_events: bool = False
    match_ids: List[int] = field(default_factory=list)


class ETLRunner:
    """Orquesta la extracción multi-recurso para una temporada.

    Parameters
    ----------
    client:
        Instancia de ``LaLigaClient``.
    repo:
        Repositorio de persistencia de payloads raw.
    season:
        Configuración de la temporada a extraer.
    sleep_between_requests:
        Segundos de espera entre peticiones para respetar rate limits.
    """

    def __init__(
        self,
        client: LaLigaClient,
        repo: RawRepository,
        season: SeasonConfig,
        sleep_between_requests: float = 0.4,
    ) -> None:
        self.client = client
        self.repo = repo
        self.season = season
        self.sleep = sleep_between_requests

    def run(self) -> None:
        """Ejecuta el pipeline completo de extracción."""
        logger.info(
            "ETL start | competition=%s season=%s subscription=%s",
            self.season.competition_slug,
            self.season.season_label,
            self.season.subscription_slug,
        )
        self._extract_global_data()
        self._extract_subscription()
        self._extract_results()
        self._extract_standing()
        self._extract_teams_stats()
        self._extract_players_stats()

        if self.season.has_match_stats or self.season.has_match_events:
            self._fan_out_matches()

        logger.info(
            "ETL complete | competition=%s season=%s",
            self.season.competition_slug,
            self.season.season_label,
        )

    # ------------------------------------------------------------------
    # Pasos individuales
    # ------------------------------------------------------------------

    def _save(self, resource: str, payload) -> None:
        self.repo.save(
            resource=resource,
            payload=payload,
            competition_slug=self.season.competition_slug,
            season_label=self.season.season_label,
        )
        time.sleep(self.sleep)

    def _extract_global_data(self) -> None:
        logger.info("Extracting global-data")
        payload = self.client.get_global_data()
        self._save("global_data", payload)

    def _extract_subscription(self) -> None:
        logger.info("Extracting subscription | %s", self.season.subscription_slug)
        payload = self.client.get_subscription(self.season.subscription_slug)
        self._save("subscription", payload)

    def _extract_results(self) -> None:
        logger.info("Extracting results | %s", self.season.subscription_slug)
        payload = self.client.get_results(self.season.subscription_slug)
        self._save("results", payload)

    def _extract_standing(self) -> None:
        logger.info("Extracting standing | %s", self.season.subscription_slug)
        payload = self.client.get_standing(self.season.subscription_slug)
        self._save("standing", payload)

    def _extract_teams_stats(self) -> None:
        logger.info("Extracting teams/stats | %s", self.season.subscription_slug)
        teams = self.client.get_all_teams_stats(self.season.subscription_slug)
        self._save("teams_stats", {"clubs": teams})

    def _extract_players_stats(self) -> None:
        logger.info("Extracting players/stats | %s", self.season.subscription_slug)
        players = self.client.get_all_players_stats(self.season.subscription_slug)
        self._save("players_stats", {"players": players})

    # ------------------------------------------------------------------
    # Fan-out por partido
    # ------------------------------------------------------------------

    def _fan_out_matches(self) -> None:
        """Extrae match stats y/o match events partido a partido."""
        if not self.season.match_ids:
            logger.warning("fan_out_matches called but match_ids list is empty")
            return

        total = len(self.season.match_ids)
        for i, match_id in enumerate(self.season.match_ids, start=1):
            logger.debug("Fan-out %d/%d match_id=%s", i, total, match_id)

            if self.season.has_match_stats:
                payload = self.client.get_match_stats(match_id)
                if payload is not None:
                    self._save(f"match_stats_{match_id}", payload)

            if self.season.has_match_events:
                payload = self.client.get_match_events(match_id)
                if payload is not None:
                    self._save(f"match_events_{match_id}", payload)


# ------------------------------------------------------------------
# Entrypoint CLI
# ------------------------------------------------------------------

def run_season(
    competition_slug: str,
    season_label: str,
    subscription_slug: str,
    subscription_key: str,
    has_match_stats: bool = False,
    has_match_events: bool = False,
    match_ids: Optional[List[int]] = None,
    db_url: Optional[str] = None,
) -> None:
    """Función de conveniencia para lanzar el ETL desde scripts externos."""
    from src.storage.repository import PostgresRawRepository

    client = LaLigaClient(subscription_key=subscription_key)
    repo = PostgresRawRepository(db_url=db_url)
    season = SeasonConfig(
        competition_slug=competition_slug,
        season_label=season_label,
        subscription_slug=subscription_slug,
        has_match_stats=has_match_stats,
        has_match_events=has_match_events,
        match_ids=match_ids or [],
    )
    ETLRunner(client=client, repo=repo, season=season).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_season(
        competition_slug="primera-division",
        season_label="2025",
        subscription_slug="laliga-easports-2025",
        subscription_key="c13c3a8e2f6b46da9c5c425cf61fab3e",
    )
