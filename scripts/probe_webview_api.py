#!/usr/bin/env python3
"""
probe_webview_api.py
--------------------
Descubre endpoints y estructura de la API webview de LaLiga.
  Base URL: https://apim.laliga.com/webview/api/web
  Subscription key diferente: ee7fcd5c543f4485ba2a48856fc7ece9

Ejecucion:
    python -m scripts.probe_webview_api --match_id 98710
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEBVIEW_BASE = "https://apim.laliga.com/webview/api/web"
WEBVIEW_KEY  = "ee7fcd5c543f4485ba2a48856fc7ece9"
PUBLIC_BASE  = "https://apim.laliga.com/public-service/api/v1"
PUBLIC_KEY   = "c13c3a8e2f6b46da9c5c425cf61fab3e"


def get(base: str, key: str, path: str, extra: dict = None) -> tuple:
    params = {"contentLanguage": "es", "subscription-key": key}
    if extra:
        params.update(extra)
    url = f"{base}/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.status_code, r.json() if r.status_code == 200 else None
    except Exception as exc:
        return None, str(exc)


def show(label, payload, max_chars=2000):
    if payload is None:
        return
    top = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    print(f"  top_keys: {top}")
    for k, v in (payload.items() if isinstance(payload, dict) else {}):
        if isinstance(v, list) and v:
            item = v[0]
            print(f"  [{k}] -> {len(v)} items, first_keys: {list(item.keys()) if isinstance(item, dict) else type(item).__name__}")
    print(f"  sample: {json.dumps(payload, ensure_ascii=False)[:max_chars]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id", type=int, default=98710)
    parser.add_argument("--slug", default="laliga-easports-2025")
    parser.add_argument("--gw_id", type=int, default=10029,
                        help="Gameweek ID para probar endpoints de jornada")
    args = parser.parse_args()

    mid = args.match_id
    slug = args.slug
    gw = args.gw_id

    candidates_webview = [
        # Detalle de partido
        (f"matches/{mid}",                  {}),
        (f"match/{mid}",                    {}),
        (f"matches/{mid}/detail",           {}),
        (f"partidazos",                     {"matchId": mid}),
        (f"matches/{mid}/score",            {}),
        (f"matches/{mid}/result",           {}),
        # Jornada
        (f"matches",                        {"gameweekId": gw, "subscriptionId": slug}),
        (f"gameweek/{gw}/matches",          {"subscriptionId": slug}),
        # Subscripcion
        (f"subscriptions/{slug}/standing",  {}),
        (f"subscriptions/{slug}",           {}),
        (f"subscriptions/{slug}/results",   {}),
    ]

    candidates_public_extra = [
        # Probar variantes en public-service con match_id correcto
        (f"matches/{mid}",                  {}),
        (f"matches/{mid}/summary",          {}),
        (f"matches/{mid}/lineups",          {}),
        (f"matches/{mid}/head2head",        {}),
    ]

    print("\n" + "="*60)
    print(f"WEBVIEW API  ({WEBVIEW_BASE})")
    print("="*60)
    for path, extra in candidates_webview:
        status, payload = get(WEBVIEW_BASE, WEBVIEW_KEY, path, extra)
        icon = "\u2705" if status == 200 else "\u274c"
        print(f"\n{icon} {status}  GET /{path}  {extra if extra else ''}")
        if payload:
            show(path, payload)

    print("\n" + "="*60)
    print(f"PUBLIC API - match extras  ({PUBLIC_BASE})")
    print("="*60)
    for path, extra in candidates_public_extra:
        status, payload = get(PUBLIC_BASE, PUBLIC_KEY, path, extra)
        icon = "\u2705" if status == 200 else "\u274c"
        print(f"\n{icon} {status}  GET /{path}")
        if payload:
            show(path, payload)


if __name__ == "__main__":
    main()
