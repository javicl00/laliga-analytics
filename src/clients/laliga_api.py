"""Cliente Python parametrizable para la API pública de LaLiga.

Endpoints verificados a 2025-04-29 (basados en HAR real del navegador):

  ✔ GET /subscriptions/{slug}
       → teams[], rounds[].gameweeks[]

  ✔ GET /subscriptions/{slug}/standing
       → standings[]

  ✔ GET /subscriptions/{slug}/teams/stats
       → team_stats[]  (stats = [{name, stat}])

  ✔ GET /subscriptions/{slug}/players/stats
       → player_stats[]  (paginado por offset)

  ✔ GET /matches?subscriptionSlug={slug}&week={week_number}&limit=100
       → matches[]  CON home_score y away_score incluidos
       NOTA: param correcto es 'subscriptionSlug' (no 'subscriptionId')
             y 'week' (numero de jornada, no gameweek_id)

  ✘ GET /matches?subscriptionId=...&gameweekId=...  → funciona pero SIN scores
  ✘ GET /subscriptions/{slug}/results             → 404
  ✘ GET /matches/{id}                             → 404
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
        logger.debug("GET %s params=%s", url, {k: v for k, v in params.items() if k != 'subscription-key'})
        r = self._session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Metadatos de temporada
    # ------------------------------------------------------------------

    def get_global_data(self) -> Any:
        return self.get("global-data", {"v3": ""})

    def get_subscription(self, subscription_slug: str) -> Any:
        """Devuelve subscription con teams[], rounds[].gameweeks[], current_gameweek."""
        return self.get(f"subscriptions/{subscription_slug}")

    def get_standing(self, subscription_slug: str) -> Any:
        """Clasificación actual. Clave raíz: 'standings'."""
        return self.get(f"subscriptions/{subscription_slug}/standing")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_teams_stats(
        self,
        subscription_slug: str,
        limit: int = 20,
        offset: int = 0,
        order_field: str = "name",
        order_type: str = "ASC",
    ) -> Any:
        """Clave raíz real: 'team_stats'. Stats = [{name, stat}]."""
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
        """Clave raíz real: 'player_stats'. Stats = [{name, stat}]."""
        return self.get(
            f"subscriptions/{subscription_slug}/players/stats",
            {"limit": limit, "offset": offset,
             "orderField": order_field, "orderType": order_type},
        )

    def get_all_players_stats(self, subscription_slug: str, page_size: int = 100) -> List[Dict]:
        """Extrae todas las páginas de player_stats usando 'total' para corte."""
        all_players: List[Dict] = []
        offset = 0
        while True:
            page = self.get_players_stats(subscription_slug, limit=page_size, offset=offset)
            players = page.get("player_stats", [])
            if not players:
                break
            all_players.extend(players)
            if offset + len(players) >= page.get("total", 0):
                break
            offset += page_size
            time.sleep(0.3)
        return all_players

    def get_all_teams_stats(self, subscription_slug: str, page_size: int = 20) -> List[Dict]:
        """Extrae todas las páginas de team_stats."""
        all_teams: List[Dict] = []
        offset = 0
        while True:
            page = self.get_teams_stats(subscription_slug, limit=page_size, offset=offset)
            teams = page.get("team_stats", [])
            if not teams:
                break
            all_teams.extend(teams)
            if offset + len(teams) >= page.get("total", 0):
                break
            offset += page_size
            time.sleep(0.3)
        return all_teams

    # ------------------------------------------------------------------
    # Partidos CON scores
    # ------------------------------------------------------------------

    def get_matches_by_week(
        self,
        subscription_slug: str,
        week: int,
        limit: int = 100,
        order_field: str = "date",
        order_type: str = "asc",
    ) -> Any:
        """Partidos de una jornada con home_score y away_score incluidos.

        PARAMETROS CORRECTOS verificados en HAR (2025-04-29):
          subscriptionSlug = slug de la temporada  (NO subscriptionId)
          week             = numero de jornada 1-38  (NO gameweekId)

        El payload incluye partidos de la competicion principal y puede
        incluir otras (Copa, Champions mismo slot). Filtrar por competition.id
        en el normalizador.
        """
        return self.get(
            "matches",
            {
                "subscriptionSlug": subscription_slug,
                "week": week,
                "limit": limit,
                "orderField": order_field,
                "orderType": order_type,
            },
        )

    def get_all_matches(
        self,
        subscription_slug: str,
        weeks: List[int],
        sleep: float = 0.4,
    ) -> Dict[int, Any]:
        """Extrae partidos para todas las jornadas (por numero de semana 1-38).

        Returns
        -------
        Dict week -> payload raw con scores incluidos.
        """
        results: Dict[int, Any] = {}
        total = len(weeks)
        for i, week in enumerate(weeks, 1):
            logger.info("Fetching matches week %d/%d (week=%s)", i, total, week)
            try:
                results[week] = self.get_matches_by_week(subscription_slug, week)
                time.sleep(sleep)
            except requests.HTTPError as exc:
                logger.warning("Error fetching week %s: %s", week, exc)
                results[week] = None
        return results

    def get_match_stats(self, match_id: int) -> Optional[Any]:
        """Stats de partido. Puede no estar disponible (devuelve None si 404)."""
        try:
            return self.get(f"matches/{match_id}/stats")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    def get_match_events(self, match_id: int) -> Optional[Any]:
        """Eventos de partido. Puede no estar disponible (devuelve None si 404)."""
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
