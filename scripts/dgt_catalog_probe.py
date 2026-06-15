#!/usr/bin/env python3
"""
DGT CDD catalogue probe.

Usage:
    python3 dgt_catalog_probe.py [collection] [limit] [bbox_W,S,E,N]

    collection  : MDT-50cm (default)
    limit       : max items to show (default 3)
    bbox        : WGS84 bbox filter, default = Algarve (-8.5,36.9,-7.85,37.2)

Examples:
    python3 dgt_catalog_probe.py MDT-50cm 5
    python3 dgt_catalog_probe.py MDT-50cm 5 -8.5,36.9,-7.85,37.2   # Algarve
    python3 dgt_catalog_probe.py MDT-50cm 5 -9.5,41.0,-8.5,42.5    # Minho
"""
import sys
import json
import requests

BASE    = "https://cdd.dgterritorio.gov.pt/dgt-be/v1"
BACKEND = "https://dgt-be.a.incd.pt:8081"   # direct — supports pagination token

def get(url, params=None):
    r = requests.get(url, params=params, timeout=90, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    })
    r.raise_for_status()
    return r.json()

def find_downloadish(obj, found, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if k.lower() in {"href", "url", "download", "asset", "assets"}:
                found.append((p, v))
            find_downloadish(v, found, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_downloadish(v, found, f"{prefix}[{i}]")

ALGARVE_BBOX = "-8.5,36.9,-7.85,37.2"   # default for reef project


def coverage_summary(collection: str) -> None:
    """Scan all pages and report latitude coverage + tile count."""
    import warnings, urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
    url = f"{BACKEND}/collections/{collection}/items"
    all_lats = []
    token_param: dict = {"limit": 500}
    while True:
        r = requests.get(url, params=token_param, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                         verify=False)
        d = r.json()
        feats = d.get("features", [])
        for f in feats:
            bb = f.get("bbox")
            if bb and len(bb) >= 4:
                all_lats.append(bb[1])   # min lat
        links = {l["rel"]: l for l in d.get("links", [])}
        if "next" in links:
            next_href = links["next"]["href"]
            token = next_href.split("token=")[-1] if "token=" in next_href else None
            if token:
                token_param = {"limit": 500, "token": token}
            else:
                break
        else:
            break
    if all_lats:
        print(f"\n{'─'*60}")
        print(f"Coverage summary for '{collection}':")
        print(f"  Total tiles : {len(all_lats)}")
        print(f"  Lat range   : {min(all_lats):.2f}°N – {max(all_lats):.2f}°N")
        algarve_n = sum(1 for lat in all_lats if lat < 38.0)
        print(f"  Algarve (<38°N): {algarve_n} tiles {'⚠ NONE' if algarve_n == 0 else '✓'}")
        print(f"{'─'*60}\n")


def main():
    collection = sys.argv[1] if len(sys.argv) > 1 else "MDT-50cm"
    limit      = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    bbox       = sys.argv[3] if len(sys.argv) > 3 else ALGARVE_BBOX

    cols = get(f"{BASE}/collections")
    collections = cols.get("data", {}).get("collections", [])
    print("\nCollections:")
    for c in collections:
        print("-", c.get("id"), "=>", c.get("titlePt") or c.get("title"))

    items_url = f"{BASE}/collections/{collection}/items"
    print(f"\nFetching items from: {items_url}")
    print(f"bbox filter: {bbox}")
    items = get(items_url, params={"limit": limit, "bbox": bbox})

    features = (
        items.get("features")
        or items.get("data", {}).get("features")
        or items.get("data", {}).get("items")
        or items.get("items")
        or []
    )

    print(f"Found {len(features)} features/items\n")
    for idx, feat in enumerate(features[:limit], 1):
        print("=" * 80)
        print("ITEM", idx)
        if isinstance(feat, dict):
            print("id:", feat.get("id"))
            print("type:", feat.get("type"))
            print("bbox:", feat.get("bbox"))
            props = feat.get("properties", {})
            for k in ["title", "datetime", "start_datetime", "end_datetime"]:
                if k in props:
                    print(f"{k}: {props[k]}")
            found = []
            find_downloadish(feat, found)
            print("\nPotential download-related fields:")
            for path, value in found[:40]:
                if isinstance(value, (str, dict, list)):
                    print("-", path, "=>", json.dumps(value, ensure_ascii=False)[:500])
        else:
            print(json.dumps(feat, ensure_ascii=False)[:2000])

    # Always show coverage summary so users know if Algarve is in the catalogue
    coverage_summary(collection)

if __name__ == "__main__":
    main()
