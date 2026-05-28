#!/usr/bin/env python3
"""
bathy_calibrator.py — IH Isobath Integration Module
=====================================================
Integrates the DGRM/IH bathymetric isobath ArcGIS REST service into the
reef imagery pipeline for:

  1. SDB Calibration     — derives m0/m1 Stumpf coefficients from IH
                           nautical chart ground-truth isobaths
  2. Zone Classification — classifies each spot as nearshore/mid/offshore
                           based on proximity to key isobaths
  3. SDB Validation      — compares Stumpf satellite depth estimates vs
                           IH chart depth at the same location
  4. Depth-weighted mask — creates a valid-observation mask (pixels within
                           the optically transparent depth window, <30m)

Service: DGRM/IH ArcGIS REST
  https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/
         Dados_entidades_externas/Batimetrica_IH/MapServer/0

Available isobaths (metres): 0, 2, 10, 20, 30, 50, 100, 200,
                              400, 500, 1000, 2000, 3000, 4000
Coverage: Caminha to Guadiana (full Portuguese coast)
CRS source: EPSG:3763 (PT-TM06), served as EPSG:4326
Max records/request: 1000 → use bbox filter always
"""

import logging
import warnings
from typing import Optional

import numpy as np
import requests
from scipy.spatial import cKDTree

from src.constants import (
    BENTHIC_ISOBATHS,
    CONTEXT_ISOBATHS,
    STUMPF_M0_DEFAULT,
    STUMPF_M1_DEFAULT,
    STUMPF_M1_LITERATURE,
    BUF_PIX,
)

log = logging.getLogger(__name__)

# ── Service constants ──────────────────────────────────────────────────────────
_IH_BASE = (
    "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/"
    "Dados_entidades_externas/Batimetrica_IH/MapServer/0"
)
_QUERY_URL = f"{_IH_BASE}/query"


# ── 1. Data fetching ───────────────────────────────────────────────────────────
def fetch_isobaths_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    depths: list[int] | None = None,
    timeout: int = 15
) -> list[dict]:
    """
    Fetch IH isobath polylines from the DGRM ArcGIS REST service for a
    given bounding box.

    Returns a list of feature dicts with keys:
        'depth'   (float)      — isobath depth in metres
        'coords'  (list)       — list of [lon, lat] coordinate pairs
        'length_deg' (float) — polyline length in degrees (Shape_Leng)

    Raises RuntimeError if the service is unreachable.
    """
    depths = depths or (BENTHIC_ISOBATHS + CONTEXT_ISOBATHS)
    depth_filter = ", ".join(str(d) for d in depths)

    params = {
        "where": f"Depth IN ({depth_filter})",
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FID,Depth,Shape_Leng",
        "returnGeometry": "true",
        "f": "json",
    }

    try:
        resp = requests.get(_QUERY_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"IH isobath service unreachable: {e}") from e

    if "error" in data:
        raise RuntimeError(f"IH service error: {data['error']}")

    features = []
    for feat in data.get("features", []):
        attrs = feat.get("attributes", {})
        geom  = feat.get("geometry", {})
        paths = geom.get("paths", [])
        depth = float(attrs.get("Depth", 0))
        length = float(attrs.get("Shape_Leng", 0.0))
        for path in paths:
            features.append({
                "depth":    depth,
                "coords":   path,          # list of [lon, lat]
                "length_deg": length,
            })

    log.info("IH isobaths fetched: %d polylines for bbox (%.3f,%.3f)→(%.3f,%.3f)",
             len(features), min_lon, min_lat, max_lon, max_lat)
    return features


# ── 2. Geometry helpers ────────────────────────────────────────────────────────
def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distance in metres between two WGS84 points."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def _build_isobath_tree(
    features: list[dict], target_depth: float
) -> tuple["cKDTree | None", "np.ndarray | None"]:
    """
    Build a cKDTree over all [lon, lat] vertices of the given isobath depth.
    Returns (None, None) if no matching features found.
    """
    coords = []
    for feat in features:
        if feat["depth"] == target_depth:
            coords.extend(feat["coords"])   # each coord is [lon, lat]
    if not coords:
        return None, None
    arr = np.array(coords, dtype=np.float64)  # shape (N, 2)
    return cKDTree(arr), arr


def min_distance_to_isobath_m(
    lon: float, lat: float,
    features: list[dict],
    target_depth: float
) -> float:
    """
    Minimum distance (metres) from point (lon, lat) to the nearest vertex
    of any isobath segment with the given target_depth.
    Uses a cKDTree for O(log n) lookup instead of an O(n) vertex scan.
    Returns np.inf if no matching isobath is found.
    """
    tree, coords = _build_isobath_tree(features, target_depth)
    if tree is None:
        return float(np.inf)
    # Find nearest vertex in degree-space, then compute haversine for accuracy
    _dist_deg, idx = tree.query([lon, lat])
    nearest = coords[idx]
    return _haversine_m(lon, lat, float(nearest[0]), float(nearest[1]))


def _stumpf_ratio(b02v: float, b03v: float, n: float = 1000.0) -> float:
    """Compute Stumpf log-ratio X = ln(n·B02) / ln(n·B03)."""
    eps = 1e-6
    return float(np.log(n * b02v + eps) / (np.log(n * b03v + eps) + eps))


def _sample_pixels_near_isobath(
    b02_arr: np.ndarray, b03_arr: np.ndarray,
    features: list[dict], target_depth: float,
    bounds_wgs84: tuple, n: float = 1000.0, buf: int = BUF_PIX
) -> list[tuple[float, float]]:
    """
    Return (depth, X) pairs for all pixels within ±buf pixels of any
    vertex on the target_depth isobath that falls inside the raster.
    Uses a set to avoid duplicating the same pixel from nearby vertices.
    """
    min_lat, min_lon, max_lat, max_lon = bounds_wgs84
    H, W = b02_arr.shape
    seen: set[tuple[int, int]] = set()
    samples: list[tuple[float, float]] = []

    for feat in features:
        if feat["depth"] != target_depth:
            continue
        for node in feat["coords"]:
            nlon, nlat = node[0], node[1]
            col0 = int((nlon - min_lon) / (max_lon - min_lon) * (W - 1))
            row0 = int((max_lat - nlat) / (max_lat - min_lat) * (H - 1))
            for dr in range(-buf, buf + 1):
                for dc in range(-buf, buf + 1):
                    r, c = row0 + dr, col0 + dc
                    if not (0 <= r < H and 0 <= c < W):
                        continue
                    if (r, c) in seen:
                        continue
                    seen.add((r, c))
                    b02v = float(b02_arr[r, c])
                    b03v = float(b03_arr[r, c])
                    if b02v <= 1e-6 or b03v <= 1e-6:
                        continue
                    X = _stumpf_ratio(b02v, b03v, n)
                    samples.append((target_depth, X))
    return samples


# ── 3. Stumpf SDB calibration ──────────────────────────────────────────────────
def calibrate_stumpf_from_isobaths(
    b02_arr: np.ndarray,
    b03_arr: np.ndarray,
    features: list[dict],
    bounds_wgs84: tuple[float, float, float, float],  # (min_lat, min_lon, max_lat, max_lon)
    n: float = 1000.0,
) -> tuple[float, float, dict]:
    """
    Derive Stumpf m0, m1 calibration coefficients using IH isobaths as
    ground-truth depth samples.

    Improvements over v1:
      - Buffer sampling: ±BUF_PIX pixels around each vertex (not just 1 pixel)
      - Single-isobath offset: if only 1 depth available, keeps m1 at the
        literature value and solves analytically for m0 (offset calibration)
      - Relaxed safety clip: allows natural Algarve fit range
    """
    # Collect samples per isobath depth using buffer sampling
    all_samples: list[tuple[float, float]] = []
    per_depth_n: dict[int, int] = {}

    for depth_m in BENTHIC_ISOBATHS:
        s = _sample_pixels_near_isobath(
            b02_arr, b03_arr, features, float(depth_m), bounds_wgs84, n
        )
        if s:
            per_depth_n[int(depth_m)] = len(s)
            all_samples.extend(s)

    if len(all_samples) < 4:
        log.warning(
            "Too few IH calibration samples (%d total) — keeping Stumpf defaults. "
            "Per-depth: %s", len(all_samples), per_depth_n
        )
        return (
            STUMPF_M0_DEFAULT, STUMPF_M1_DEFAULT,
            {"n_samples": len(all_samples), "calibrated": False,
             "rmse_m": None, "isobaths_used": [],
             "per_depth_n": per_depth_n,
             "reason": "insufficient_samples (<4)"}
        )

    depths_arr = np.array([s[0] for s in all_samples])
    X_arr      = np.array([s[1] for s in all_samples])
    distinct_depths = sorted(set(depths_arr.tolist()))
    isobaths_used   = [int(d) for d in distinct_depths]

    # ── Strategy A: Full OLS regression (≥2 distinct isobath depths) ──────────
    if len(distinct_depths) >= 2:
        m1_raw, m0_raw = np.polyfit(X_arr, depths_arr, 1)
        depth_pred_raw = m1_raw * X_arr + m0_raw
        rmse_raw = float(np.sqrt(np.mean((depth_pred_raw - depths_arr) ** 2)))

        # Physical bounds: Algarve oligotrophic clear-water regime
        # m0 range: −60 to +5 m  |  m1 range: 10 to 70
        m0 = float(np.clip(m0_raw, -60.0, 5.0))
        m1 = float(np.clip(m1_raw,  10.0, 70.0))
        clipped = (m0 != m0_raw or m1 != m1_raw)
        if clipped:
            log.warning(
                "OLS fit hit safety bounds (raw m0=%.2f m1=%.2f) → "
                "clipped to m0=%.2f m1=%.2f", m0_raw, m1_raw, m0, m1
            )

        depth_pred = m1 * X_arr + m0
        rmse = float(np.sqrt(np.mean((depth_pred - depths_arr) ** 2)))
        method = "ols_regression"

    # ── Strategy B: Single-isobath offset calibration ─────────────────────────
    # Fix m1 at literature value (physically grounded) and solve for m0
    # analytically: m0 = mean(depth) − m1 · mean(X)
    # This is far better than using generic defaults (which have the wrong offset).
    else:
        d0 = distinct_depths[0]
        m1 = STUMPF_M1_LITERATURE
        m0 = float(np.mean(depths_arr) - m1 * np.mean(X_arr))
        # Soft bound: only clip if truly unphysical
        m0 = float(np.clip(m0, -80.0, 10.0))
        depth_pred = m1 * X_arr + m0
        rmse = float(np.sqrt(np.mean((depth_pred - depths_arr) ** 2)))
        method = f"offset_calibration_single_isobath({int(d0)}m)"
        log.info(
            "Single-isobath offset calibration on %gm: "
            "m1_fixed=%.2f  m0_solved=%.3f  RMSE=%.2fm  n=%d",
            d0, m1, m0, rmse, len(all_samples)
        )

    log.info(
        "Stumpf calibration [%s]: m0=%.3f  m1=%.3f  RMSE=%.3fm  "
        "n=%d samples  per_depth=%s",
        method, m0, m1, rmse, len(all_samples), per_depth_n
    )

    return (m0, m1, {
        "n_samples":     len(all_samples),
        "calibrated":    True,
        "method":        method,
        "m0":            round(m0, 4),
        "m1":            round(m1, 4),
        "rmse_m":        round(rmse, 4),
        "isobaths_used": isobaths_used,
        "per_depth_n":   per_depth_n,
    })


# ── 4. Zone classification ─────────────────────────────────────────────────────
def classify_benthic_zone(
    lon: float, lat: float,
    features: list[dict]
) -> dict:
    """
    Classify a reef observation point by depth zone using IH isobaths.

    Returns a dict with:
        zone         : 'very_shallow' | 'shallow_reef' | 'mid_depth' | 'offshore'
        dist_10m_m   : distance to 10m isobath in metres
        dist_20m_m   : distance to 20m isobath in metres
        dist_30m_m   : distance to 30m isobath in metres
        optically_viable : True if within useful S2 depth window (<30m)
        note         : human-readable description
    """
    d10 = min_distance_to_isobath_m(lon, lat, features, 10.0)
    d20 = min_distance_to_isobath_m(lon, lat, features, 20.0)
    d30 = min_distance_to_isobath_m(lon, lat, features, 30.0)
    d50 = min_distance_to_isobath_m(lon, lat, features, 50.0)

    def fmt(d):
        return round(d, 1) if not np.isinf(d) else None

    # Use the nearest isobath to infer approximate zone.
    # Only use a distance value if that isobath actually exists in the dataset
    # (i.e. not inf). This avoids misclassifying nearshore spots when the
    # 30m or 50m line simply falls outside the bbox.
    available = {}
    if not np.isinf(d10): available["10m"] = d10
    if not np.isinf(d20): available["20m"] = d20
    if not np.isinf(d30): available["30m"] = d30
    if not np.isinf(d50): available["50m"] = d50

    nearest_iso = min(available, key=lambda k: available[k]) if available else "unknown"

    # Classification rules — deepest available isobath that is very close
    # takes priority; if no isobath is within meaningful range, classify offshore
    if d10 < 200:
        zone = "very_shallow"
        note = "Within 200m of the 10m isobath — very shallow reef (optically clear)"
        optically_viable = True
    elif d20 < 500:
        zone = "shallow_reef"
        note = "Within 500m of the 20m isobath — prime shallow reef zone"
        optically_viable = True
    elif d10 < 1500 or d20 < 1500:
        # Near 10m or 20m even if >500m — still inside the optical window
        zone = "nearshore_mid"
        note = f"Within 1.5km of shallow isobaths (10m={fmt(d10)}m, 20m={fmt(d20)}m) — viable"
        optically_viable = True
    elif not np.isinf(d30) and d30 < 1000:
        zone = "mid_depth"
        note = "Near 30m isobath — at the edge of Sentinel-2 optical window"
        optically_viable = True
    elif not np.isinf(d50) and d50 < 500:
        zone = "offshore"
        note = "Closest isobath is 50m — low SNR expected from Sentinel-2"
        optically_viable = False
    else:
        # All available isobaths are far — likely offshore but 30/50m not in bbox
        # Fall back to nearest known isobath distance to decide
        min_known_d = min(available.values()) if available else np.inf
        if min_known_d < 2000:
            zone = "nearshore_mid"
            note = f"All isobaths >1km away but nearest ({nearest_iso}) at {round(min_known_d)}m — probably mid-shelf"
            optically_viable = True
        else:
            zone = "offshore"
            note = "All isobaths far from observation point — likely offshore or deep"
            optically_viable = False

    return {
        "zone":             zone,
        "nearest_isobath":  nearest_iso,
        "dist_10m_m":       fmt(d10),
        "dist_20m_m":       fmt(d20),
        "dist_30m_m":       fmt(d30),
        "dist_50m_m":       fmt(d50),
        "optically_viable": optically_viable,
        "note":             note,
    }


# ── 5. SDB validation ──────────────────────────────────────────────────────────
def validate_sdb_vs_chart(
    sdb_map: np.ndarray,
    features: list[dict],
    bounds_wgs84: tuple[float, float, float, float],
    isobaths_to_check: list[int] | None = None
) -> dict:
    """
    Validate the Stumpf SDB depth map against IH chart isobaths.

    For each isobath depth, extract SDB pixel values at chart locations
    and compute mean error (bias) and RMSE.

    Returns dict with per-isobath validation stats and overall bias.
    """
    isobaths_to_check = isobaths_to_check or BENTHIC_ISOBATHS
    min_lat, min_lon, max_lat, max_lon = bounds_wgs84
    H, W = sdb_map.shape

    results = {}
    all_errors = []

    for target_depth in isobaths_to_check:
        errors = []
        for feat in features:
            if feat["depth"] != float(target_depth):
                continue
            for node in feat["coords"]:
                nlon, nlat = node[0], node[1]
                col = int((nlon - min_lon) / (max_lon - min_lon) * (W - 1))
                row = int((max_lat - nlat) / (max_lat - min_lat) * (H - 1))
                if not (0 <= row < H and 0 <= col < W):
                    continue
                sdb_val = float(sdb_map[row, col])
                if sdb_val > 0:
                    errors.append(sdb_val - target_depth)

        if errors:
            errors_arr = np.array(errors)
            results[f"{target_depth}m"] = {
                "n_samples":    len(errors),
                "bias_m":       round(float(np.mean(errors_arr)), 2),
                "rmse_m":       round(float(np.sqrt(np.mean(errors_arr**2))), 2),
                "mae_m":        round(float(np.mean(np.abs(errors_arr))), 2),
            }
            all_errors.extend(errors)
        else:
            results[f"{target_depth}m"] = {"n_samples": 0, "bias_m": None,
                                           "rmse_m": None, "mae_m": None}

    overall = {}
    if all_errors:
        ae = np.array(all_errors)
        overall = {
            "overall_bias_m": round(float(np.mean(ae)), 2),
            "overall_rmse_m": round(float(np.sqrt(np.mean(ae**2))), 2),
            "overall_mae_m":  round(float(np.mean(np.abs(ae))), 2),
            "n_total":        len(all_errors),
        }
        log.info(
            "SDB vs IH chart | overall bias=%.2fm  RMSE=%.2fm  MAE=%.2fm  n=%d",
            overall["overall_bias_m"], overall["overall_rmse_m"],
            overall["overall_mae_m"], overall["n_total"]
        )

    return {"per_isobath": results, "overall": overall}


# ── 6. High-level convenience function ────────────────────────────────────────
def run_bathy_integration(
    lat: float,
    lon: float,
    buffer_m: float = 3000.0,
    b02_arr: Optional[np.ndarray] = None,
    b03_arr: Optional[np.ndarray] = None,
    sdb_map: Optional[np.ndarray] = None,
    bounds_wgs84: Optional[tuple] = None,
) -> dict:
    """
    Full integration pipeline for one observation point.

    Parameters
    ----------
    lat, lon      : centre of the reef observation window
    buffer_m      : half-width of the search bbox in metres (~0.027° per km)
    b02_arr       : B02 reflectance raster (for calibration)
    b03_arr       : B03 reflectance raster (for calibration)
    sdb_map       : Stumpf SDB output raster (for validation)
    bounds_wgs84  : (min_lat, min_lon, max_lat, max_lon) of the rasters

    Returns
    -------
    dict with keys:
        isobaths_available  : list of depth values found in bbox
        zone                : zone classification dict
        calibration         : Stumpf m0/m1 calibration dict (if rasters provided)
        validation          : SDB vs chart validation dict (if sdb_map provided)
        recommended_m0      : float
        recommended_m1      : float
    """
    # Convert buffer_m to rough degree offset (at lat≈37°N, 1°≈111km)
    deg_buf = buffer_m / 111_000.0
    min_lon = lon - deg_buf
    max_lon = lon + deg_buf
    min_lat = lat - deg_buf
    max_lat = lat + deg_buf

    result = {
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "isobaths_available": [],
        "zone": {},
        "calibration": {},
        "validation": {},
        "recommended_m0": STUMPF_M0_DEFAULT,
        "recommended_m1": STUMPF_M1_DEFAULT,
    }

    try:
        features = fetch_isobaths_for_bbox(min_lon, min_lat, max_lon, max_lat)
    except RuntimeError as e:
        log.error("Bathy integration failed: %s", e)
        result["error"] = str(e)
        return result

    result["isobaths_available"] = sorted(set(f["depth"] for f in features))

    # Zone classification
    result["zone"] = classify_benthic_zone(lon, lat, features)

    # Stumpf calibration (requires rasters)
    if b02_arr is not None and b03_arr is not None and bounds_wgs84 is not None:
        m0, m1, cal_diag = calibrate_stumpf_from_isobaths(
            b02_arr, b03_arr, features, bounds_wgs84
        )
        result["calibration"] = cal_diag
        result["recommended_m0"] = m0
        result["recommended_m1"] = m1
    else:
        result["calibration"] = {"calibrated": False,
                                  "note": "No rasters provided — defaults kept"}

    # SDB validation (requires sdb_map + raster bounds)
    if sdb_map is not None and bounds_wgs84 is not None:
        result["validation"] = validate_sdb_vs_chart(
            sdb_map, features, bounds_wgs84
        )

    return result


# ── CLI demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Test against the main Pedra do Alto spot
    print("\n=== Pedra do Alto ===")
    r1 = run_bathy_integration(lat=37.0636, lon=-8.2193)
    print(json.dumps(r1, indent=2))

    print("\n=== Target (East Reef) ===")
    r2 = run_bathy_integration(lat=37.0468, lon=-7.6603)
    print(json.dumps(r2, indent=2))

    print("\n=== Galé Spot ===")
    r3 = run_bathy_integration(lat=37.0562, lon=-8.2296)
    print(json.dumps(r3, indent=2))
