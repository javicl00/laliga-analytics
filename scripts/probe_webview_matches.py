#!/usr/bin/env python3
"""
probe_webview_matches.py
------------------------
Descubre el endpoint de partidos/scores en la API webview de LaLiga.

Ejecucion:
    python -m scripts.probe_webview_matches
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEBVIEW_BASE = "https://apim.laliga.com/webview/api/web"
WEBVIEW_KEY  = "ee7fcd5c543f4485ba2a48856fc7ece9"

SLUG   = "laliga-easports-2025"
GW_ID  = 10029   # Jornada 1 temporada 2025
GW_WK  = 1       # week number
MATCH  = 98710


def get(path: str, extra: dict = None):
    params = {"contentLanguage": "es", "subscription-key": WEBVIEW_KEY}
    if extra:
        params.update(extra)
    url = f"{WEBVIEW_BASE}/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.status_code, r.json() if r.status_code == 200 else None
    except Exception as e:
        return None, str(e)


def show(payload, max_chars=2500):
    if not payload:
        return
    top = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    print(f"    top_keys: {top}")
    for k, v in (payload.items() if isinstance(payload, dict) else {}).items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print(f"    [{k}] -> {len(v)} items | first_keys: {list(v[0].keys())}")
    print(f"    sample: {json.dumps(payload, ensure_ascii=False)[:max_chars]}")


CANDIDATES = [
    # ---- Partidos por jornada (webview) ----
    (f"subscriptions/{SLUG}/gameweeks/{GW_ID}/matches",  {}),
    (f"subscriptions/{SLUG}/gameweeks/{GW_WK}/matches",  {}),
    (f"subscriptions/{SLUG}/matches",                    {"gameweek": GW_ID}),
    (f"subscriptions/{SLUG}/matches",                    {"week": GW_WK}),
    (f"subscriptions/{SLUG}/matches",                    {}),
    (f"subscriptions/{SLUG}/results",                    {"gameweekId": GW_ID}),
    (f"subscriptions/{SLUG}/results",                    {"week": GW_WK}),
    (f"subscriptions/{SLUG}/calendar",                   {}),
    (f"subscriptions/{SLUG}/schedule",                   {}),
    (f"subscriptions/{SLUG}/rounds",                     {}),
    # ---- Partidos por ID ----
    (f"matches/{MATCH}",                                 {}),
    (f"matches",                                         {"matchId": MATCH}),
    (f"matches",                                         {"id": MATCH}),
    (f"match/{MATCH}/detail",                            {}),
    (f"match",                                           {"matchId": MATCH}),
    # ---- Endpoints globales de partidos ----
    (f"matches",                                         {"subscriptionId": SLUG}),
    (f"matches",                                         {"subscriptionId": SLUG, "gameweekId": GW_ID}),
    (f"matches",                                         {"subscriptionId": SLUG, "week": GW_WK}),
    # ---- Endpoints conocidos del HAR ----
    (f"games",                                           {"subscriptionId": SLUG, "gameweekId": GW_ID}),
    (f"games/{MATCH}",                                   {}),
    (f"game/{MATCH}",                                    {}),
    # ---- Jornada completa ----
    (f"gameweeks/{GW_ID}",                               {"subscriptionId": SLUG}),
    (f"gameweeks/{GW_ID}/matches",                       {"subscriptionId": SLUG}),
]


def main():
    print(f"\nProbing WEBVIEW API for matches | slug={SLUG} gw_id={GW_ID} match={MATCH}\n")
    found = []
    for path, extra in CANDIDATES:
        status, payload = get(path, extra)
        icon = "\u2705" if status == 200 else "\u274c"
        label = f"{path}  {extra if extra else ''}"
        print(f"{icon} {status}  GET /{label}")
        if payload:
            show(payload)
            found.append((path, extra, payload))

    print(f"\n{'='*50}")
    print(f"Found {len(found)} working endpoints")
    for p, e, _ in found:
        print(f"  -> /{p}  {e}")


if __name__ == "__main__":
    main()
