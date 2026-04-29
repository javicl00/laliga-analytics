"""Cliente Python parametrizable para la API pública de LaLiga.

Cambios respecto a v1 (basados en estructura real verificada):
- /results → 404. Los partidos están en /gameweek/{id}/matches o /calendar.
- team_stats usa clave 'team_stats' (no 'clubs' ni 'teams').
- player_stats usa clave 'player_stats' (no 'players').
- standing usa clave 'standings'.
- stats es lista [{name, stat}], no dict plano.
- Añadido: get_gameweek_matches, get_calendar para obtener partidos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://apim.laliga.com/public-service/api/v1"
DEFAULT_SUBSCRIPTION_KEY = "c13c3a8e2f6b46da9c5c425cf61fab3e"


class LaLigaClient:
    """Cliente HTTP para la API pública de LaLiga."""

    def __init__(
        self,
        subscription_key: str = DEFAULT_SUBSCRIPTION_KEY,
        language: str = "es",
        base_url: str = BASE_URL,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> None:
        self.subscription_key = subscription_key
        self.language = language
        self.base_url = base_url.rstrip("/")
        self._session = self._build_session(max_retries, backoff_factor)

    def _build_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Any:
        params = dict(params or {})
        params["contentLanguage"] = self.language
        params["subscription-key"] = self.subscription_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        logger.debug("GET %s", url)
        r = self._session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Endpoints verificados
    # ------------------------------------------------------------------

    def get_global_data(self) -> Any:
        return self.get("global-data", {"v3": ""})

    def get_subscription(self, subscription_slug: str) -> Any:
        """Devuelve subscription con teams[], rounds[].gameweeks[], current_gameweek."""
        return self.get(f"subscriptions/{subscription_slug}")

    def get_standing(self, subscription_slug: str) -> Any:
        """Clasificación actual. Claves: total, standings[]."""
        return self.get(f"subscriptions/{subscription_slug}/standing")

    def get_teams_stats(
        self,
        subscription_slug: str,
        limit: int = 20,
        offset: int = 0,
        order_field: str = "name",
        order_type: str = "ASC",
    ) -> Any:
        """Stats de equipos. Clave raíz real: 'team_stats' (no 'clubs')."""
        return self.get(
            f"subscriptions/{subscription_slug}/teams/stats",
            {"limit": limit, "offset": offset,
             "orderField": order_field, "orderType": order_type},
        )

    def get_players_stats(
        self,
        subscription_slug: str,
        limit: int = 100,
        offset: int = 0,
        order_field: str = "name",
        order_type: str = "ASC",
    ) -> Any:
        """Stats de jugadores. Clave raíz real: 'player_stats' (no 'players')."""
        return self.get(
            f"subscriptions/{subscription_slug}/players/stats",
            {"limit": limit, "offset": offset,
             "orderField": order_field, "orderType": order_type},
        )

    def get_all_players_stats(self, subscription_slug: str, page_size: int = 100) -> List[Dict]:
        """Pagina sobre player_stats hasta agotar."""
        all_players: List[Dict] = []
        offset = 0
        while True:
            page = self.get_players_stats(subscription_slug, limit=page_size, offset=offset)
            players = page.get("player_stats", [])  # clave real verificada
            if not players:
                break
            all_players.extend(players)
            if len(players) < page_size:
                break
            offset += page_size
            time.sleep(0.3)
        return all_players

    def get_all_teams_stats(self, subscription_slug: str, page_size: int = 20) -> List[Dict]:
        """Pagina sobre team_stats (20 equipos, normalmente 1 página)."""
        all_teams: List[Dict] = []
        offset = 0
        while True:
            page = self.get_teams_stats(subscription_slug, limit=page_size, offset=offset)
            teams = page.get("team_stats", [])  # clave real verificada
            if not teams:
                break
            all_teams.extend(teams)
            if len(teams) < page_size:
                break
            offset += page_size
            time.sleep(0.3)
        return all_teams

    # ------------------------------------------------------------------
    # Endpoints de partidos (estructura a confirmar, pendiente sondeo)
    # ------------------------------------------------------------------

    def get_gameweek_matches(self, subscription_slug: str, gameweek_id: int) -> Optional[Any]:
        """Intenta obtener los partidos de una jornada concreta.

        Prueba múltiples variantes de URL hasta encontrar la correcta.
        Devuelve None si ninguna funciona.
        """
        candidates = [
            f"subscriptions/{subscription_slug}/gameweek/{gameweek_id}/matches",
            f"subscriptions/{subscription_slug}/matches?gameweekId={gameweek_id}",
            f"subscriptions/{subscription_slug}/calendar?gameweekId={gameweek_id}",
        ]
        for path in candidates:
            try:
                result = self.get(path)
                logger.info("get_gameweek_matches: working path = %s", path)
                return result
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.debug("404 for path: %s", path)
                    continue
                raise
        logger.warning("get_gameweek_matches: no working path found for gw=%s", gameweek_id)
        return None

    def get_calendar(self, subscription_slug: str) -> Optional[Any]:
        """Intenta obtener el calendario completo de la temporada."""
        candidates = [
            f"subscriptions/{subscription_slug}/calendar",
            f"subscriptions/{subscription_slug}/schedule",
            f"subscriptions/{subscription_slug}/fixtures",
        ]
        for path in candidates:
            try:
                result = self.get(path)
                logger.info("get_calendar: working path = %s", path)
                return result
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
        return None

    # ------------------------------------------------------------------
    # Fan-out por partido
    # ------------------------------------------------------------------

    def get_match_stats(self, match_id: int) -> Optional[Any]:
        try:
            return self.get(f"matches/{match_id}/stats")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    def get_match_events(self, match_id: int) -> Optional[Any]:
        try:
            return self.get(f"matches/{match_id}/events")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    @staticmethod
    def request_hash(payload: Any) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()
