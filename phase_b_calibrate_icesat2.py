"""Phase B: ICESat-2 calibration helpers for satellite-derived bathymetry.

Public API consumed by tests/test_phase_b.py:
  fit_calibration      — OLS linear fit: Z_true = a * Z_SDB + b
  apply_calibration    — apply a/b to a depth raster
  match_icesat2_to_sdb — spatial match of ICESat-2 points to SDB pixels
  _OUTPUT_DIR          — default output directory Path
"""

import math
from pathlib import Path

import numpy as np

_OUTPUT_DIR = Path("outputs/phase_b_calibration")

_MIN_SAMPLES = 4
_SLOPE_MIN, _SLOPE_MAX = 0.5, 2.0


def fit_calibration(icesat2: np.ndarray, sdb: np.ndarray) -> dict:
    """Fit Z_calibrated = a * Z_SDB + b via OLS on matched depth pairs.

    Requires at least _MIN_SAMPLES pairs; returns identity coefficients when
    fewer are available.  Slope is clipped to [0.5, 2.0].
    """
    n = len(icesat2)
    if n < _MIN_SAMPLES:
        return {
            "calibrated": False,
            "n_samples": n,
            "a": 1.0,
            "b": 0.0,
            "rmse_m": None,
            "bias_m": None,
            "reason": f"insufficient samples: {n} < {_MIN_SAMPLES}",
        }

    a_raw, b_raw = np.polyfit(sdb, icesat2, 1)
    a = float(np.clip(a_raw, _SLOPE_MIN, _SLOPE_MAX))
    b = float(b_raw)

    residuals = icesat2 - (a * sdb + b)
    return {
        "calibrated": True,
        "n_samples": n,
        "a": a,
        "b": b,
        "rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "bias_m": float(np.mean(residuals)),
        "reason": None,
    }


def apply_calibration(sdb_arr: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Z_calibrated = a * Z_SDB + b to a depth raster.

    Zero pixels (nodata/land) are preserved; calibrated depths are clipped
    to ≥ 0 so no negative values are introduced.
    """
    zero_mask = sdb_arr == 0.0
    result = a * sdb_arr + b
    result = np.clip(result, 0.0, None)
    result[zero_mask] = 0.0
    return result.astype(sdb_arr.dtype)


def match_icesat2_to_sdb(
    icesat2_pts: np.ndarray,
    sdb_arr: np.ndarray,
    bounds_wgs84: tuple,
    max_distance_m: float = 30.0,
) -> tuple:
    """Match ICESat-2 photon returns to SDB raster pixels.

    For each point the nearest pixel centre is found; a match is accepted
    when the pixel is non-zero (water) and within max_distance_m.

    Args:
        icesat2_pts:   (N, 3) array — columns [lon, lat, depth_m].
        sdb_arr:       (H, W) float depth raster.
        bounds_wgs84:  (minx, miny, maxx, maxy) in decimal degrees WGS84.
        max_distance_m: acceptance radius in metres.

    Returns:
        (icesat2_depths, sdb_depths) — matched 1-D arrays.
    """
    minx, miny, maxx, maxy = bounds_wgs84
    H, W = sdb_arr.shape
    dx = (maxx - minx) / W
    dy = (maxy - miny) / H

    icesat2_out: list = []
    sdb_out: list = []

    for lon, lat, depth in icesat2_pts:
        if not (minx <= lon <= maxx and miny <= lat <= maxy):
            continue

        col = min(int((lon - minx) / dx), W - 1)
        row = min(int((maxy - lat) / dy), H - 1)

        sdb_val = float(sdb_arr[row, col])
        if sdb_val == 0.0:
            continue

        px_lon = minx + (col + 0.5) * dx
        px_lat = maxy - (row + 0.5) * dy

        if _dist_m(lon, lat, px_lon, px_lat) <= max_distance_m:
            icesat2_out.append(depth)
            sdb_out.append(sdb_val)

    return np.array(icesat2_out), np.array(sdb_out)


def _dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Planar approximation of WGS84 distance in metres."""
    lat_m = 111111.0
    lon_m = 111111.0 * abs(math.cos(math.radians((lat1 + lat2) / 2.0)))
    return math.sqrt(((lon2 - lon1) * lon_m) ** 2 + ((lat2 - lat1) * lat_m) ** 2)
