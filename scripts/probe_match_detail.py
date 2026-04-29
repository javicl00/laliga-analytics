#!/usr/bin/env python3
"""
probe_match_detail.py
---------------------
Verifica la estructura de GET /matches/{id} para confirmar dónde están los goles.

Ejecución:
    python -m scripts.probe_match_detail --match_id 100710
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id", type=int, required=True, help="ID de un partido terminado")
    parser.add_argument("--key", default=DEFAULT_SUBSCRIPTION_KEY)
    args = parser.parse_args()

    client = LaLigaClient(subscription_key=args.key)

    print(f"\n=== GET /matches/{args.match_id} ===")
    detail = client.get_match_detail(args.match_id)
    if detail is None:
        print("404 - No encontrado")
        return
    print(f"top_keys: {list(detail.keys()) if isinstance(detail, dict) else type(detail).__name__}")
    print(json.dumps(detail, indent=2, ensure_ascii=False)[:3000])

    print(f"\n=== GET /matches/{args.match_id}/stats ===")
    stats = client.get_match_stats(args.match_id)
    if stats:
        print(f"top_keys: {list(stats.keys())}")
        print(json.dumps(stats, indent=2, ensure_ascii=False)[:1500])
    else:
        print("404 o no disponible")


if __name__ == "__main__":
    main()
