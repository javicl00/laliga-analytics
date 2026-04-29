"""Repositorios de persistencia para raw payloads."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


class RawRepository:
    """Interfaz base para repositorios de payloads raw."""

    def save(
        self,
        resource: str,
        payload: Any,
        competition_slug: str,
        season_label: str,
        request_url: str = "",
    ) -> None:
        raise NotImplementedError


class InMemoryRawRepository(RawRepository):
    """Repositorio en memoria para tests y entornos sin BD."""

    def __init__(self) -> None:
        self.records: list = []

    def save(self, resource, payload, competition_slug, season_label, request_url="") -> None:
        self.records.append({
            "resource": resource,
            "competition_slug": competition_slug,
            "season_label": season_label,
            "request_url": request_url,
            "payload": payload,
            "payload_hash": _hash_payload(payload),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug("InMemory save | resource=%s season=%s", resource, season_label)


class PostgresRawRepository(RawRepository):
    """Repositorio persistente en PostgreSQL via psycopg2."""

    def __init__(self, db_url: Optional[str] = None, conn=None) -> None:
        """
        Acepta un db_url o una conexión psycopg2 ya establecida.
        Si se pasa db_url, la conexión se crea de forma lazy.
        """
        self._db_url = db_url
        self._conn = conn

    def _get_conn(self):
        if self._conn is None:
            import psycopg2
            self._conn = psycopg2.connect(self._db_url)
        return self._conn

    def save(self, resource, payload, competition_slug, season_label, request_url="") -> None:
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload_hash = _hash_payload(payload)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into raw_payloads
                  (competition_slug, season_label, resource, request_url, payload, payload_hash)
                values (%s, %s, %s, %s, %s::jsonb, %s)
                on conflict do nothing
                """,
                [competition_slug, season_label, resource, request_url, payload_text, payload_hash],
            )
        conn.commit()
        logger.info("Saved raw payload | resource=%s season=%s hash=%s", resource, season_label, payload_hash[:8])
