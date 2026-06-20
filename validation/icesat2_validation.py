"""
icesat2_validation.py — ICESat-2 ATL08 validation of SDB depth maps per candidate.

Wraps src.icesat2_validation (single-scene) with:
  - Per-candidate batch validation for all 8 Algarve reef sites
  - Local photon cache (validation/cache_photon.py) — avoids repeat API calls
  - Folium interactive map of co-located SDB vs ICESat-2 points
  - validation_report.json with RMSE / MAE / bias per candidate
  - validation_map.html (Leaflet/Folium)

Target: <2 min total for 8 candidates on CPU.

Usage:
    python validation/icesat2_validation.py --depth-map outputs/.../sdb_depth_map.tif
    python validation/icesat2_validation.py --candidates outputs/reef_candidates.geojson
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.icesat2_validation import run_icesat2_validation
from validation.cache_photon import PhotonCache

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Per-candidate validation
# ---------------------------------------------------------------------------

def validate_candidate(
    candidate: dict[str, Any],
    depth_map_path: str | Path,
    output_dir: str | Path,
    cache: PhotonCache,
) -> dict[str, Any]:
    """
    Run ICESat-2 validation for one reef candidate.

    Args:
        candidate:      GeoJSON Feature dict with properties: name, lat, lon, date.
        depth_map_path: Path to the SDB depth raster.
        output_dir:     Per-candidate output directory.
        cache:          PhotonCache instance.

    Returns:
        Validation result dict (RMSE, MAE, bias, n_points, status).
    """
    props    = candidate.get("properties", {})
    name     = props.get("name", "unknown")
    date_str = props.get("date", "")
    out_dir  = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = run_icesat2_validation(
        depth_map_path=str(depth_map_path),
        output_dir=str(out_dir),
        scene_date=date_str,
    )
    elapsed = time.perf_counter() - t0

    if result is None:
        return {"candidate": name, "status": "skipped", "elapsed_s": round(elapsed, 1)}

    return {
        "candidate": name,
        "status":    "ok",
        "rmse_m":    result.get("rmse_m"),
        "mae_m":     result.get("mae_m"),
        "bias_m":    result.get("bias_m"),
        "n_points":  result.get("n_co_located"),
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Folium map
# ---------------------------------------------------------------------------

def _build_folium_map(
    results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import folium
    except ImportError:
        log.warning("folium not installed — skipping validation_map.html")
        return

    center_lat = 37.07
    center_lon = -8.40
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

    result_by_name = {r["candidate"]: r for r in results}

    for feat in candidates:
        props = feat.get("properties", {})
        name  = props.get("name", "?")
        geom  = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        lon, lat = (coords[0], coords[1]) if len(coords) >= 2 else (None, None)
        if lat is None:
            continue

        r = result_by_name.get(name, {})
        status = r.get("status", "unknown")
        color  = "green" if status == "ok" else "gray"

        popup_lines = [f"<b>{name}</b>"]
        if status == "ok":
            popup_lines += [
                f"RMSE: {r.get('rmse_m', '?'):.2f} m",
                f"MAE:  {r.get('mae_m',  '?'):.2f} m",
                f"Bias: {r.get('bias_m', '?'):+.2f} m",
                f"N:    {r.get('n_points', '?')}",
            ]
        else:
            popup_lines.append(f"Status: {status}")

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup("<br>".join(popup_lines), max_width=200),
            tooltip=name,
        ).add_to(m)

    m.save(str(output_path))
    log.info("Validation map saved: %s", output_path)


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_all_candidates(
    candidates_geojson: str | Path,
    depth_map_path: str | Path,
    output_dir: str | Path = "outputs/validation",
) -> dict[str, Any]:
    """
    Validate all reef candidates against ICESat-2 ATL08 data.

    Args:
        candidates_geojson: Path to reef_candidates.geojson.
        depth_map_path:     Path to the SDB depth raster.
        output_dir:         Root output directory for per-candidate results.

    Returns:
        Summary dict with per-candidate results and aggregate statistics.
    """
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache    = PhotonCache(cache_dir="data/cache_icesat2")

    candidates: list[dict] = []
    try:
        data = json.loads(Path(candidates_geojson).read_text())
        candidates = data.get("features", [])
    except Exception as exc:
        log.error("Cannot load candidates GeoJSON: %s", exc)
        return {"status": "error", "message": str(exc)}

    log.info("Validating %d candidates…", len(candidates))
    t0_total = time.perf_counter()
    results  = []

    for feat in candidates:
        res = validate_candidate(feat, depth_map_path, out_dir / "candidates", cache)
        results.append(res)
        log.info("  %s — %s  %.1fs", res["candidate"], res["status"], res["elapsed_s"])

    # Aggregate stats over successful validations
    ok = [r for r in results if r["status"] == "ok"]
    agg: dict[str, Any] = {}
    if ok:
        agg["mean_rmse_m"] = round(float(np.mean([r["rmse_m"] for r in ok])), 3)
        agg["mean_mae_m"]  = round(float(np.mean([r["mae_m"]  for r in ok])), 3)
        agg["mean_bias_m"] = round(float(np.mean([r["bias_m"] for r in ok])), 3)
        agg["n_validated"] = len(ok)

    total_elapsed = round(time.perf_counter() - t0_total, 1)
    summary = {
        "status":        "ok",
        "total_elapsed_s": total_elapsed,
        "n_candidates":  len(candidates),
        "aggregate":     agg,
        "results":       results,
    }

    # Write JSON report
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    log.info("Validation report: %s  (%.1f s total)", report_path, total_elapsed)

    # Build Folium map
    _build_folium_map(results, candidates, out_dir / "validation_map.html")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description="ICESat-2 validation for reef candidates")
    p.add_argument("--candidates",  default="outputs/reef_candidates.geojson",
                   help="GeoJSON file with reef candidates")
    p.add_argument("--depth-map",   default=None,
                   help="SDB depth raster TIF (optional override)")
    p.add_argument("--output-dir",  default="outputs/validation")
    args = p.parse_args()

    depth_map = args.depth_map or "outputs/sdb_depth_map.tif"
    summary = validate_all_candidates(args.candidates, depth_map, args.output_dir)

    agg = summary.get("aggregate", {})
    if agg:
        print(f"\nAggregate validation ({agg['n_validated']} sites):")
        print(f"  Mean RMSE : {agg['mean_rmse_m']:.2f} m")
        print(f"  Mean MAE  : {agg['mean_mae_m']:.2f} m")
        print(f"  Mean bias : {agg['mean_bias_m']:+.2f} m")
    print(f"\nTotal time: {summary['total_elapsed_s']:.1f} s")


if __name__ == "__main__":
    _cli()
