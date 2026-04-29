#!/usr/bin/env python3
"""
explore_endpoints.py
--------------------
Script de exploración de la API real de LaLiga.

Ejecución:
    python -m scripts.explore_endpoints
    python -m scripts.explore_endpoints --season laliga-easports-2024
    python -m scripts.explore_endpoints --output output/explore_2025.json

Para cada endpoint clave guarda:
  - Estructura de claves top-level
  - Tipo y ejemplo del primer elemento de cada lista
  - Número de registros
  - Nombres de stats disponibles (para teams/stats y players/stats)

No escribe nada en base de datos. Solo lectura.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clients.laliga_api import LaLigaClient, DEFAULT_SUBSCRIPTION_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de análisis estructural
# ---------------------------------------------------------------------------

def schema_of(obj: Any, depth: int = 2) -> Any:
    """Devuelve la 'forma' de un objeto JSON hasta `depth` niveles."""
    if depth == 0:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: schema_of(v, depth - 1) for k, v in list(obj.items())[:30]}
    if isinstance(obj, list):
        if not obj:
            return []
        return [schema_of(obj[0], depth - 1), f"... ({len(obj)} items)"]
    return type(obj).__name__


def extract_stat_names(payload: Any) -> List[str]:
    """Extrae todos los nombres de stat de un payload de teams/stats o players/stats."""
    stat_names = set()
    items = (
        payload.get("clubs")
        or payload.get("teams")
        or payload.get("players")
        or payload.get("data")
        or []
    )
    for item in items[:5]:  # Suficiente con los primeros
        stats = item.get("statistics") or item.get("stats") or []
        if isinstance(stats, list):
            for s in stats:
                if isinstance(s, dict) and "name" in s:
                    stat_names.add(s["name"])
                elif isinstance(s, dict):
                    stat_names.update(s.keys())
        elif isinstance(stats, dict):
            stat_names.update(stats.keys())
        # También busca directamente en el objeto
        for k, v in item.items():
            if k.lower() in ("statistics", "stats", "statistic"):
                continue
            if isinstance(v, (int, float)) and k not in ("id", "team_id", "player_id"):
                stat_names.add(k)
    return sorted(stat_names)


def count_records(payload: Any) -> Dict[str, int]:
    """Cuenta registros en las claves principales del payload."""
    result = {}
    if not isinstance(payload, dict):
        return {"root": 1}
    for k, v in payload.items():
        if isinstance(v, list):
            result[k] = len(v)
    return result


def first_item(payload: Any) -> Any:
    """Devuelve el primer elemento de la lista principal del payload."""
    if not isinstance(payload, dict):
        return payload
    for k, v in payload.items():
        if isinstance(v, list) and v:
            return v[0]
    return None


def summarize_match_structure(payload: Any) -> Dict:
    """Analiza estructura específica de results con partidos."""
    matches = (
        payload.get("matches")
        or payload.get("rounds")
        or payload.get("results")
        or []
    )
    # Puede que los partidos estén anidados en rounds
    flat_matches = []
    for m in matches:
        if isinstance(m, dict) and "matches" in m:
            flat_matches.extend(m["matches"])
        elif isinstance(m, dict):
            flat_matches.append(m)

    if not flat_matches:
        return {"structure": schema_of(payload, 2), "count": 0}

    sample = flat_matches[0]
    return {
        "total_matches": len(flat_matches),
        "match_keys": list(sample.keys()) if isinstance(sample, dict) else [],
        "sample_match": schema_of(sample, 1),
        "status_values": list({m.get("status", "?") for m in flat_matches[:50] if isinstance(m, dict)}),
    }


def summarize_standing_structure(payload: Any) -> Dict:
    """Analiza estructura de standing."""
    rows = (
        payload.get("standings")
        or payload.get("standing")
        or payload.get("data")
        or []
    )
    if not rows:
        return {"structure": schema_of(payload, 2), "count": 0}
    return {
        "total_teams": len(rows),
        "row_keys": list(rows[0].keys()) if isinstance(rows[0], dict) else [],
        "sample_row": schema_of(rows[0], 1),
    }


# ---------------------------------------------------------------------------
# Exploración principal
# ---------------------------------------------------------------------------

class EndpointExplorer:
    def __init__(self, client: LaLigaClient, subscription_slug: str) -> None:
        self.client = client
        self.slug = subscription_slug
        self.report: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Explorando endpoints para slug: %s", self.slug)
        logger.info("=" * 60)

        self._explore_global_data()
        self._explore_subscription()
        self._explore_results()
        self._explore_standing()
        self._explore_teams_stats()
        self._explore_players_stats()

        return self.report

    def _save(self, key: str, summary: Dict) -> None:
        self.report[key] = summary
        logger.info("[%s] %s", key, json.dumps(summary, ensure_ascii=False, indent=2)[:800])

    def _explore_global_data(self) -> None:
        logger.info("\n--- global-data ---")
        try:
            payload = self.client.get_global_data()
            summary = {
                "top_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                "schema": schema_of(payload, 2),
            }
            self._save("global_data", summary)
        except Exception as e:
            self._save("global_data", {"error": str(e)})

    def _explore_subscription(self) -> None:
        logger.info("\n--- subscription: %s ---", self.slug)
        try:
            payload = self.client.get_subscription(self.slug)
            summary = {
                "top_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                "record_counts": count_records(payload),
                "schema": schema_of(payload, 2),
            }
            self._save("subscription", summary)
        except Exception as e:
            self._save("subscription", {"error": str(e)})

    def _explore_results(self) -> None:
        logger.info("\n--- results ---")
        try:
            payload = self.client.get_results(self.slug)
            summary = {
                "top_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                "record_counts": count_records(payload),
                "match_structure": summarize_match_structure(payload),
                "schema_depth2": schema_of(payload, 2),
            }
            self._save("results", summary)
        except Exception as e:
            self._save("results", {"error": str(e)})

    def _explore_standing(self) -> None:
        logger.info("\n--- standing ---")
        try:
            payload = self.client.get_standing(self.slug)
            summary = {
                "top_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                "record_counts": count_records(payload),
                "standing_structure": summarize_standing_structure(payload),
                "schema_depth2": schema_of(payload, 2),
            }
            self._save("standing", summary)
        except Exception as e:
            self._save("standing", {"error": str(e)})

    def _explore_teams_stats(self) -> None:
        logger.info("\n--- teams/stats ---")
        try:
            payload = self.client.get_teams_stats(self.slug, limit=5)
            stat_names = extract_stat_names(payload)
            summary = {
                "top_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                "record_counts": count_records(payload),
                "stat_names": stat_names,
                "stat_count": len(stat_names),
                "first_item_keys": list(first_item(payload).keys()) if first_item(payload) else [],
                "schema_depth2": schema_of(payload, 2),
            }
            self._save("teams_stats", summary)
        except Exception as e:
            self._save("teams_stats", {"error": str(e)})

    def _explore_players_stats(self) -> None:
        logger.info("\n--- players/stats ---")
        try:
            payload = self.client.get_players_stats(self.slug, limit=5)
            stat_names = extract_stat_names(payload)
            summary = {
                "top_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                "record_counts": count_records(payload),
                "stat_names": stat_names,
                "stat_count": len(stat_names),
                "first_item_keys": list(first_item(payload).keys()) if first_item(payload) else [],
                "schema_depth2": schema_of(payload, 2),
            }
            self._save("players_stats", summary)
        except Exception as e:
            self._save("players_stats", {"error": str(e)})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Explora estructura real de la API de LaLiga")
    parser.add_argument(
        "--season",
        default="laliga-easports-2025",
        help="Subscription slug (ej: laliga-easports-2025)",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_SUBSCRIPTION_KEY,
        help="Subscription key de la API",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta donde guardar el JSON de exploración (opcional)",
    )
    args = parser.parse_args()

    client = LaLigaClient(subscription_key=args.key)
    explorer = EndpointExplorer(client=client, subscription_slug=args.season)
    report = explorer.run()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("\nInforme guardado en: %s", out)
    else:
        print("\n" + "=" * 60)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
