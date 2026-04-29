#!/usr/bin/env python3
"""
probe_matches_endpoint.py
-------------------------
Descubre el endpoint real para obtener partidos de una jornada.

Ejecución:
    python -m scripts.probe_matches_endpoint
    python -m scripts.probe_matches_endpoint --season laliga-easports-2025 --gw 10029
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clients.laliga_api import LaLigaClient, DEFAULT_SUBSCRIPTION_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CANDIDATE_PATHS = [
    "subscriptions/{slug}/gameweek/{gw_id}/matches",
    "subscriptions/{slug}/matches?gameweekId={gw_id}",
    "subscriptions/{slug}/calendar",
    "subscriptions/{slug}/schedule",
    "subscriptions/{slug}/fixtures",
    "subscriptions/{slug}/gameweek/{gw_id}",
    "subscriptions/{slug}/round/{gw_id}/matches",
    "subscriptions/{slug}/results2025",
    "subscriptions/{slug}/results",
    "matches?subscriptionId={slug}&gameweekId={gw_id}",
]


def probe(client: LaLigaClient, slug: str, gw_id: int) -> None:
    print(f"\nProbing match endpoints for slug={slug} gw_id={gw_id}\n")
    for template in CANDIDATE_PATHS:
        path = template.format(slug=slug, gw_id=gw_id)
        try:
            result = client.get(path)
            print(f"\n✅ FOUND: GET /{path}")
            print(f"   top_keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, list) and v:
                        print(f"   [{k}] → {len(v)} items, first keys: {list(v[0].keys()) if isinstance(v[0], dict) else type(v[0]).__name__}")
                        print(f"   Sample: {json.dumps(v[0], ensure_ascii=False)[:500]}")
                        break
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', '?')
            print(f"   ❌ {status}  GET /{path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="laliga-easports-2025")
    parser.add_argument("--gw", type=int, default=10029, help="Gameweek ID (ej: 10029 = J1 2025)")
    parser.add_argument("--key", default=DEFAULT_SUBSCRIPTION_KEY)
    args = parser.parse_args()
    client = LaLigaClient(subscription_key=args.key)
    probe(client, args.season, args.gw)


if __name__ == "__main__":
    main()
