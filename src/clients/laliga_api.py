"""Cliente Python parametrizable para la API pública de LaLiga.

Soporta paginación automática, reintentos con backoff exponencial,
caché por hash de request y logging estructurado por recurso.
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
    """Cliente HTTP para la API pública de LaLiga.

    Parameters
    ----------
    subscription_key:
        Clave de suscripción de la API.
    language:
        Idioma del contenido devuelto (``es`` por defecto).
    base_url:
        URL base de la API. Usar el valor por defecto en producción.
    max_retries:
        Número máximo de reintentos ante errores 429/5xx.
    backoff_factor:
        Factor de espera exponencial entre reintentos.
    """

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

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Any:
        """Ejecuta un GET y devuelve el payload JSON."""
        params = dict(params or {})
        params["contentLanguage"] = self.language
        params["subscription-key"] = self.subscription_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        logger.debug("GET %s params=%s", url, {k: v for k, v in params.items() if k != 'subscription-key'})
        response = self._session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Helpers especializados
    # ------------------------------------------------------------------

    def get_global_data(self) -> Any:
        """Obtiene la configuración global de competiciones y recursos."""
        return self.get("global-data", {"v3": ""})

    def get_subscription(self, subscription_slug: str) -> Any:
        """Metadatos de suscripción: equipos, jornadas y estructura."""
        return self.get(f"subscriptions/{subscription_slug}")

    def get_results(self, subscription_slug: str) -> Any:
        """Resultados de todos los partidos de la temporada."""
        return self.get(f"subscriptions/{subscription_slug}/results")

    def get_standing(self, subscription_slug: str) -> Any:
        """Clasificación actual de la temporada."""
        return self.get(f"subscriptions/{subscription_slug}/standing")

    def get_teams_stats(
        self,
        subscription_slug: str,
        limit: int = 100,
        offset: int = 0,
        order_field: str = "name",
        order_type: str = "ASC",
    ) -> Any:
        """Estadísticas de equipos con paginación."""
        return self.get(
            f"subscriptions/{subscription_slug}/teams/stats",
            {
                "limit": limit,
                "offset": offset,
                "orderField": order_field,
                "orderType": order_type,
            },
        )

    def get_players_stats(
        self,
        subscription_slug: str,
        limit: int = 100,
        offset: int = 0,
        order_field: str = "name",
        order_type: str = "ASC",
    ) -> Any:
        """Estadísticas de jugadores con paginación."""
        return self.get(
            f"subscriptions/{subscription_slug}/players/stats",
            {
                "limit": limit,
                "offset": offset,
                "orderField": order_field,
                "orderType": order_type,
            },
        )

    def get_all_players_stats(
        self,
        subscription_slug: str,
        page_size: int = 100,
    ) -> List[Dict]:
        """Extrae todas las páginas de estadísticas de jugadores."""
        all_players: List[Dict] = []
        offset = 0
        while True:
            page = self.get_players_stats(subscription_slug, limit=page_size, offset=offset)
            players = page.get("players") or page.get("data") or []
            if not players:
                break
            all_players.extend(players)
            if len(players) < page_size:
                break
            offset += page_size
            time.sleep(0.3)
        return all_players

    def get_all_teams_stats(
        self,
        subscription_slug: str,
        page_size: int = 100,
    ) -> List[Dict]:
        """Extrae todas las páginas de estadísticas de equipos."""
        all_teams: List[Dict] = []
        offset = 0
        while True:
            page = self.get_teams_stats(subscription_slug, limit=page_size, offset=offset)
            teams = page.get("clubs") or page.get("teams") or page.get("data") or []
            if not teams:
                break
            all_teams.extend(teams)
            if len(teams) < page_size:
                break
            offset += page_size
            time.sleep(0.3)
        return all_teams

    # ------------------------------------------------------------------
    # Fan-out por partido (match stats / match events)
    # ------------------------------------------------------------------

    def get_match_stats(self, match_id: int) -> Optional[Any]:
        """Estadísticas de un partido concreto. Devuelve None si no existe."""
        try:
            return self.get(f"matches/{match_id}/stats")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("match_stats not found for match_id=%s", match_id)
                return None
            raise

    def get_match_events(self, match_id: int) -> Optional[Any]:
        """Eventos de un partido concreto. Devuelve None si no existe."""
        try:
            return self.get(f"matches/{match_id}/events")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("match_events not found for match_id=%s", match_id)
                return None
            raise

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def request_hash(payload: Any) -> str:
        """SHA-256 del payload serializado para detección de duplicados."""
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()
