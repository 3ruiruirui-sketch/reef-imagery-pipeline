"""
icesat2_validation.py — ICESat-2 ATL08 validation of SDB depth maps.

Fetches ATL08 photon-level data from the OpenAltimetry public API and
compares sampled elevations against the pipeline's Stumpf SDB depth raster,
reporting RMSE, bias, and point count.

Public API — no credentials required.

Usage (as function called from the pipeline):
    from src.icesat2_validation import run_icesat2_validation

    report = run_icesat2_validation(
        depth_map_path="outputs/.../sdb_depth_map.tif",
        output_dir="outputs/.../validation",
    )
    # report is also written to output_dir/validation_report.json

The function returns None and skips gracefully if:
    - OpenAltimetry is unreachable
    - No ATL08 photons are found in the bounding box and date range
    - The depth raster cannot be opened
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger(__name__)

_OA_URL = "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2/atl08"
_TIMEOUT = 15  # seconds per request
_SEARCH_DAYS = 180  # ±days around the scene date to search for passes
_TRACK_IDS = list(range(1, 15))  # ICESat-2 has 3 beams × 2 sides; try IDs 1-14

# Maximum allowed depth for SDB comparison (optical limit)
_SDB_MAX_DEPTH_M = 40.0
# ATL08 elevation sign convention: positive above WGS84 ellipsoid.
# For shallow coastal water the seafloor returns arrive below sea level.
# We treat negative ATL08 elevations (below sea surface) as water depth.
_MAX_ATL08_DEPTH_M = 35.0  # discard photons deeper than this (noise/deep water)


# ── OpenAltimetry fetch ───────────────────────────────────────────────────────


def _fetch_atl08_for_date(
    bbox: tuple[float, float, float, float],
    query_date: str,
    track_id: int,
) -> list[dict]:
    """
    Query OpenAltimetry ATL08 for a single date and track.
    Returns a list of photon dicts, or [] on error / no data.
    """
    params = {
        "minx": bbox[0],
        "miny": bbox[1],
        "maxx": bbox[2],
        "maxy": bbox[3],
        "date": query_date,
        "trackId": track_id,
        "outputFormat": "json",
    }
    try:
        r = requests.get(_OA_URL, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        photons = []
        for series in data.get("series", []):
            photons.extend(series.get("data", []))
        return photons
    except Exception as e:
        log.debug("ATL08 fetch error date=%s track=%d: %s", query_date, track_id, e)
        return []


def _search_atl08(
    bbox: tuple[float, float, float, float],
    scene_date: str,
    search_days: int = _SEARCH_DAYS,
) -> list[dict]:
    """
    Search a ±search_days window around scene_date for ATL08 passes.

    ICESat-2 has a 91-day exact repeat cycle. Searching ±180 days gives two
    possible repeat passes. We try every date in the window at the track IDs
    most likely to cross the Algarve (ICESat-2 beam spacing ~3.3 km).

    Returns all collected photon dicts.
    """
    try:
        centre = date.fromisoformat(scene_date)
    except ValueError:
        log.warning("Invalid scene date %r — skipping ICESat-2 validation.", scene_date)
        return []

    all_photons: list[dict] = []
    dates_to_try = [
        (centre + timedelta(days=d)).isoformat() for d in range(-search_days, search_days + 1, 7)  # weekly sampling
    ]

    for query_date in dates_to_try:
        for track_id in _TRACK_IDS:
            pts = _fetch_atl08_for_date(bbox, query_date, track_id)
            if pts:
                log.info(
                    "ATL08: %d photons on %s track=%d over bbox",
                    len(pts),
                    query_date,
                    track_id,
                )
                all_photons.extend(pts)
                # Once we have enough points, stop searching this date
                if len(all_photons) >= 500:
                    break
        if len(all_photons) >= 500:
            break

    return all_photons


# ── SDB raster sampling ───────────────────────────────────────────────────────


def _sample_sdb_at_points(
    depth_map_path: str | Path,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample SDB depth raster at (lon, lat) coordinates.

    Returns an array of depth values in metres (NaN where outside raster
    extent or nodata).
    """
    import rasterio
    from pyproj import Transformer

    depths = np.full(len(lons), np.nan)
    try:
        with rasterio.open(str(depth_map_path)) as src:
            tf = Transformer.from_crs("EPSG:4326", src.crs.to_epsg(), always_xy=True)
            xs, ys = tf.transform(lons, lats)
            rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
            arr = src.read(1)
            nodata = src.nodata
            H, W = arr.shape
            for i, (r, c) in enumerate(zip(rows, cols)):
                if 0 <= r < H and 0 <= c < W:
                    v = float(arr[r, c])
                    if nodata is None or not math.isclose(v, nodata, rel_tol=1e-5):
                        depths[i] = v
    except Exception as e:
        log.warning("SDB raster sampling failed: %s", e)
    return depths


# ── Validation report ─────────────────────────────────────────────────────────


def run_icesat2_validation(
    depth_map_path: str | Path,
    output_dir: str | Path,
    scene_date: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    search_days: int = _SEARCH_DAYS,
) -> dict | None:
    """
    Validate SDB depth map against ICESat-2 ATL08 photon data.

    1. Derives bbox and date from depth_map_path if not supplied.
    2. Searches OpenAltimetry ATL08 for photons in ±search_days window.
    3. Filters photons to shallow coastal range (0–35 m below sea surface).
    4. Samples the SDB raster at each photon location.
    5. Computes RMSE, bias, Pearson-r, and N points.
    6. Writes validation_report.json to output_dir.

    Args:
        depth_map_path: path to the SDB depth GeoTIFF produced by the pipeline.
        output_dir:     directory where validation_report.json will be written.
        scene_date:     YYYY-MM-DD of the Sentinel-2 acquisition (used to centre
                        the search window). Derived from raster metadata if None.
        bbox:           (minx, miny, maxx, maxy) in WGS84. Derived from raster
                        extent if None.
        search_days:    half-width of date search window in days.

    Returns:
        dict with validation metrics, or None if skipped.
    """
    depth_map_path = Path(depth_map_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_report.json"

    if not depth_map_path.exists():
        log.warning("ICESat-2 validation skipped — depth map not found: %s", depth_map_path)
        return None

    # ── Derive bbox and date from raster if not provided ──────────────────────
    if bbox is None or scene_date is None:
        try:
            import rasterio
            from pyproj import Transformer

            with rasterio.open(str(depth_map_path)) as src:
                if bbox is None:
                    b = src.bounds
                    tf = Transformer.from_crs(src.crs.to_epsg(), 4326, always_xy=True)
                    w, s = tf.transform(b.left, b.bottom)
                    e, n = tf.transform(b.right, b.top)
                    bbox = (w, s, e, n)
                if scene_date is None:
                    # Extract date from filename: supports YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD
                    import re

                    m = re.search(r"(20\d{2})([-_]?)(\d{2})\2(\d{2})", depth_map_path.stem)
                    if m:
                        scene_date = f"{m.group(1)}-{m.group(3)}-{m.group(4)}"
        except Exception as e:
            log.warning("Could not read depth raster metadata: %s", e)
            return None

    if scene_date is None:
        scene_date = date.today().isoformat()
        log.warning("Scene date unknown — using today (%s) as search centre.", scene_date)

    log.info(
        "ICESat-2 validation: bbox=%s  date=%s  ±%d days",
        bbox,
        scene_date,
        search_days,
    )

    # ── Fetch ATL08 photons ───────────────────────────────────────────────────
    photons = _search_atl08(bbox, scene_date, search_days)

    if not photons:
        log.info("ICESat-2 validation skipped — no ATL08 photons found in search window.")
        report = {
            "status": "skipped",
            "reason": "no ATL08 photons in bbox/date range",
            "depth_map": str(depth_map_path),
            "scene_date": scene_date,
            "bbox": bbox,
            "search_days": search_days,
            "n_photons": 0,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # ── Extract lat/lon/elevation from photon records ─────────────────────────
    # OpenAltimetry ATL08 record shape: {"lat": ..., "lon": ..., "h": ..., ...}
    lats, lons, elevs = [], [], []
    for p in photons:
        lat = p.get("lat") or p.get("latitude")
        lon = p.get("lon") or p.get("longitude")
        h = p.get("h") or p.get("height") or p.get("elevation")
        if lat is not None and lon is not None and h is not None:
            lats.append(float(lat))
            lons.append(float(lon))
            elevs.append(float(h))

    if not lats:
        log.info("ICESat-2 validation skipped — photon records had no lat/lon/h fields.")
        report = {
            "status": "skipped",
            "reason": "photon records missing lat/lon/h fields",
            "n_photons_raw": len(photons),
            "sample_keys": list(photons[0].keys()) if photons else [],
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    lats = np.array(lats)
    lons = np.array(lons)
    elevs = np.array(elevs)

    # Filter: keep only photons with negative elevation (below sea surface)
    # and within the expected shallow-water depth range.
    depth_mask = (elevs < 0) & (elevs > -_MAX_ATL08_DEPTH_M)
    icesat_depth = -elevs[depth_mask]  # convert to positive depth in metres
    lats_f = lats[depth_mask]
    lons_f = lons[depth_mask]

    log.info(
        "ATL08: %d total photons, %d in shallow-water depth range (0–%.0f m)",
        len(elevs),
        int(depth_mask.sum()),
        _MAX_ATL08_DEPTH_M,
    )

    if len(icesat_depth) < 5:
        log.info(
            "ICESat-2 validation skipped — only %d shallow-water photons (need ≥5).",
            len(icesat_depth),
        )
        report = {
            "status": "skipped",
            "reason": f"only {len(icesat_depth)} shallow-water photons (need ≥5)",
            "n_photons_total": len(elevs),
            "n_photons_shallow": int(depth_mask.sum()),
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # ── Sample SDB raster ─────────────────────────────────────────────────────
    sdb_depths = _sample_sdb_at_points(depth_map_path, lons_f, lats_f)

    # SDB rasters may store depth as NEGATIVE (below sea surface), while
    # icesat_depth above is POSITIVE (= -elevs). Comparing the two raw produces a
    # spurious bias of ~2x the true depth. Normalize SDB to positive depth using
    # the median sign of the valid samples so residuals are physically meaningful.
    _finite = np.isfinite(sdb_depths)
    if _finite.any() and np.median(sdb_depths[_finite]) < 0:
        sdb_depths = -sdb_depths
        log.info("SDB samples have negative median — flipped to positive-depth convention.")

    valid = np.isfinite(sdb_depths) & np.isfinite(icesat_depth)
    n_valid = int(valid.sum())

    if n_valid < 5:
        log.info(
            "ICESat-2 validation skipped — only %d co-located points after raster sampling.",
            n_valid,
        )
        report = {
            "status": "skipped",
            "reason": f"only {n_valid} co-located points (need ≥5)",
            "n_photons_shallow": len(icesat_depth),
            "n_colocated": n_valid,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # ── Metrics ───────────────────────────────────────────────────────────────
    icesat_v = icesat_depth[valid]
    sdb_v = sdb_depths[valid]
    residuals = sdb_v - icesat_v

    rmse = float(np.sqrt(np.mean(residuals**2)))
    bias = float(np.mean(residuals))
    mae = float(np.mean(np.abs(residuals)))

    # Pearson r (requires ≥2 distinct values)
    if icesat_v.std() > 1e-6 and sdb_v.std() > 1e-6:
        r_val = float(np.corrcoef(icesat_v, sdb_v)[0, 1])
    else:
        r_val = float("nan")

    report = {
        "status": "ok",
        "depth_map": str(depth_map_path),
        "scene_date": scene_date,
        "bbox": bbox,
        "search_days": search_days,
        "n_photons_total": len(elevs),
        "n_photons_shallow": len(icesat_depth),
        "n_colocated": n_valid,
        "rmse_m": round(rmse, 4),
        "bias_m": round(bias, 4),
        "mae_m": round(mae, 4),
        "pearson_r": round(r_val, 4) if not math.isnan(r_val) else None,
        "icesat2_depth_mean_m": round(float(icesat_v.mean()), 3),
        "icesat2_depth_std_m": round(float(icesat_v.std()), 3),
        "sdb_depth_mean_m": round(float(sdb_v.mean()), 3),
        "sdb_depth_std_m": round(float(sdb_v.std()), 3),
        "interpretation": (
            f"SDB vs ICESat-2: RMSE={rmse:.2f}m, bias={bias:+.2f}m "
            f"({'over' if bias > 0 else 'under'}estimates depth), N={n_valid}"
        ),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(
        "ICESat-2 validation complete — RMSE=%.2fm  bias=%+.2fm  N=%d  → %s",
        rmse,
        bias,
        n_valid,
        report_path,
    )
    return report
