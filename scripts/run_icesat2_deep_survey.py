#!/usr/bin/env python3
"""
ICESat-2 ATL08 deep-zone reef survey — Algarve coast 15–30 m.

Uses the OpenAltimetry public API (no credentials needed) to collect
ATL08 seafloor photon returns in the 15–30 m depth band across the full
Algarve coast (Cabo de Santa Maria → Cabo do Carvoeiro).

Clusters nearby photons into candidate reef patches and saves validated
GeoJSON ready for the dashboard /api/candidates?layer=icesat2_deep route.

Usage:
    python scripts/run_icesat2_deep_survey.py
"""

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Full Algarve extent
BBOX = (-8.50, 36.90, -7.85, 37.20)   # W, S, E, N

DEPTH_MIN_M =  15.0   # shallow cutoff
DEPTH_MAX_M =  30.0   # deep cutoff (optical/ATL08 limit)

# OpenAltimetry endpoint
_OA_URL  = "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2/atl08"
_TIMEOUT = 20

# Search window: last 2 years in 14-day steps (covers multiple 91-day repeat cycles)
_CENTRE_DATE   = date(2025, 6, 1)
_SEARCH_DAYS   = 365
_STEP_DAYS     = 14
_TRACK_IDS     = list(range(1, 15))
_TARGET_PHOTONS = 300

OUT_DIR = _PROJECT_ROOT / "outputs" / "icesat2_deep_survey"


def fetch_photons(bbox, query_date: str, track_id: int) -> list[dict]:
    params = {
        "minx": bbox[0], "miny": bbox[1],
        "maxx": bbox[2], "maxy": bbox[3],
        "date": query_date,
        "trackId": track_id,
        "outputFormat": "json",
    }
    try:
        r = requests.get(_OA_URL, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        pts = []
        for series in data.get("series", []):
            pts.extend(series.get("data", []))
        return pts
    except Exception as exc:
        log.debug("ATL08 error date=%s track=%d: %s", query_date, track_id, exc)
        return []


def search_all_photons(bbox) -> list[dict]:
    """Search ±SEARCH_DAYS around CENTRE_DATE in STEP_DAYS increments."""
    all_pts: list[dict] = []
    dates = [
        (_CENTRE_DATE + timedelta(days=d)).isoformat()
        for d in range(-_SEARCH_DAYS, _SEARCH_DAYS + 1, _STEP_DAYS)
    ]
    log.info("Searching %d date windows × %d tracks …", len(dates), len(_TRACK_IDS))
    for qdate in dates:
        for tid in _TRACK_IDS:
            pts = fetch_photons(bbox, qdate, tid)
            if pts:
                log.info("  %s track=%d → %d photons", qdate, tid, len(pts))
                all_pts.extend(pts)
            if len(all_pts) >= _TARGET_PHOTONS:
                log.info("Reached target of %d photons — stopping early", _TARGET_PHOTONS)
                return all_pts
    return all_pts


def filter_depth(photons: list[dict]) -> list[dict]:
    """Keep only photons in the 15–30 m depth band (negative elevation = below sea level)."""
    kept = []
    for p in photons:
        try:
            elev = float(p.get("h", p.get("elevation", p.get("z", 0))))
        except (TypeError, ValueError):
            continue
        # ATL08: negative elevation = seafloor below sea surface
        depth_m = abs(elev) if elev < 0 else elev  # handle both sign conventions
        if DEPTH_MIN_M <= depth_m <= DEPTH_MAX_M:
            kept.append({**p, "_depth_m": depth_m})
    return kept


def cluster_to_candidates(photons: list[dict], cluster_deg: float = 0.02) -> list[dict]:
    """
    Simple grid-cell clustering: group photons into 0.02° × 0.02° cells
    (~2 km × 1.5 km at 37°N). Each cell with ≥2 photons becomes a candidate.
    Returns list of candidate dicts with centroid, depth stats, photon count.
    """
    from collections import defaultdict

    cells: dict[tuple, list] = defaultdict(list)
    for p in photons:
        try:
            lon = float(p.get("lon", p.get("longitude", p.get("x", 0))))
            lat = float(p.get("lat", p.get("latitude",  p.get("y", 0))))
        except (TypeError, ValueError):
            continue
        cell_key = (round(lon / cluster_deg) * cluster_deg,
                    round(lat / cluster_deg) * cluster_deg)
        cells[cell_key].append(p)

    candidates = []
    for (clon, clat), pts in cells.items():
        if len(pts) < 2:
            continue
        depths = [p["_depth_m"] for p in pts]
        candidates.append({
            "lon": clon,
            "lat": clat,
            "photon_count": len(pts),
            "depth_mean_m": float(np.mean(depths)),
            "depth_std_m":  float(np.std(depths)),
            "depth_min_m":  float(np.min(depths)),
            "depth_max_m":  float(np.max(depths)),
        })

    log.info("Clustered %d photons → %d candidates (≥2 photons/cell)", len(photons), len(candidates))
    return candidates


def score_candidates(candidates: list[dict]) -> list[dict]:
    """Assign confidence score based on photon count and depth consistency."""
    for c in candidates:
        score = 40
        notes = []

        n = c["photon_count"]
        if n >= 10:   score += 25; notes.append(f"Dense returns ({n} photons)")
        elif n >= 5:  score += 15; notes.append(f"Moderate returns ({n} photons)")
        else:         score += 5;  notes.append(f"Sparse returns ({n} photons)")

        std = c["depth_std_m"]
        if std < 1.5:  score += 20; notes.append(f"Consistent depth (±{std:.1f}m)")
        elif std < 3:  score += 10; notes.append(f"Moderate depth variation (±{std:.1f}m)")
        else:          score -= 10; notes.append(f"High depth variation (±{std:.1f}m) — slope or noise")

        dm = c["depth_mean_m"]
        notes.append(f"Mean depth {dm:.1f}m")

        score = max(0, min(100, score))
        cls = ("HIGH CONFIDENCE — Likely real reef" if score >= 70
               else "MODERATE — Needs verification" if score >= 50
               else "LOW — Probably noise/artifact" if score >= 30
               else "REJECT — Likely false positive")

        c["confidence_score"]  = score
        c["validation_class"]  = cls
        c["validation_notes"]  = " | ".join(notes)
        c["source"]            = "ICESat-2 ATL08"
        c["depth_zone"]        = "15–30 m"

    return candidates


def to_geojson(candidates: list[dict], out_path: Path) -> None:
    """Write candidates as GeoJSON point features."""
    features = []
    for i, c in enumerate(candidates):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [c["lon"], c["lat"]],
            },
            "properties": {
                "id": i,
                "confidence_score":  c["confidence_score"],
                "validation_class":  c["validation_class"],
                "validation_notes":  c["validation_notes"],
                "photon_count":      c["photon_count"],
                "depth_mean_m":      round(c["depth_mean_m"], 1),
                "depth_std_m":       round(c["depth_std_m"],  1),
                "source":            c["source"],
                "depth_zone":        c["depth_zone"],
            },
        })

    fc = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(fc, indent=2))
    log.info("Saved %d features → %s", len(features), out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = OUT_DIR / "atl08_raw_photons.json"

    # Use cached photons if available
    if cache_path.exists():
        log.info("Loading cached photons from %s", cache_path)
        all_photons = json.loads(cache_path.read_text())
    else:
        log.info("=== Fetching ATL08 photons (bbox: %.2f,%.2f → %.2f,%.2f) ===", *BBOX)
        all_photons = search_all_photons(BBOX)
        cache_path.write_text(json.dumps(all_photons))
        log.info("Cached %d raw photons → %s", len(all_photons), cache_path)

    if not all_photons:
        log.warning("No photons found — OpenAltimetry may be unreachable or no passes in window")
        # Write empty GeoJSON so dashboard route doesn't 404
        to_geojson([], OUT_DIR / "candidates_icesat2_deep.geojson")
        return

    log.info("Total raw photons: %d", len(all_photons))

    deep_photons = filter_depth(all_photons)
    log.info("Photons in 15–30 m band: %d / %d", len(deep_photons), len(all_photons))

    if not deep_photons:
        log.warning("No photons in 15–30 m depth band")
        to_geojson([], OUT_DIR / "candidates_icesat2_deep.geojson")
        return

    candidates = cluster_to_candidates(deep_photons)
    candidates = score_candidates(candidates)

    high     = sum(1 for c in candidates if c["confidence_score"] >= 70)
    moderate = sum(1 for c in candidates if 50 <= c["confidence_score"] < 70)
    low      = sum(1 for c in candidates if c["confidence_score"] < 50)

    log.info("=== RESULTS ===")
    log.info("Total candidates : %d", len(candidates))
    log.info("HIGH  (≥70)      : %d", high)
    log.info("MODERATE (50-69) : %d", moderate)
    log.info("LOW/REJECT (<50) : %d", low)

    out_path = OUT_DIR / "candidates_icesat2_deep.geojson"
    to_geojson(candidates, out_path)

    log.info("=== DONE — %s ===", out_path)


if __name__ == "__main__":
    main()
