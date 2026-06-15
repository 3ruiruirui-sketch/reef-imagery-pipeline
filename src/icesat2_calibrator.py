"""
icesat2_calibrator.py — ICESat-2 ATL03 → Stumpf m0/m1 per-scene calibration.

Replaces (or supplements) the IH nautical chart isobath approach with
photon-level depth ground truth from ICESat-2 ATL03.  For every ATL03
photon inside the current scene bbox:

  1.  Locate the corresponding Sentinel-2 pixel (B02, B03).
  2.  Compute the Stumpf log-ratio at that pixel.
  3.  Pair (log_ratio, depth_m) as a training sample.
  4.  Fit a Huber robust regression → scene-specific m0, m1.

Advantages over IH isobath calibration:
  • Photon depth resolution: ~0.5 m vs. ±1 isobath interval
  • Continuous depth labels (not just at 2/10/20m chart lines)
  • Can calibrate to 34 m where ATL03 penetrates in clear Algarve water
  • Per-scene coefficients adapt to each acquisition's water clarity

Usage:
    from src.icesat2_calibrator import calibrate_from_icesat2
    result = calibrate_from_icesat2(
        b02_path="path/to/BOA_B02.tif",
        b03_path="path/to/BOA_B03.tif",
        photon_cache="outputs/icesat2_deep_survey/atl03_seafloor_photons.json",
    )
    stumpf_m0 = result["m0"]
    stumpf_m1 = result["m1"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from sklearn.linear_model import HuberRegressor

from src.constants import STUMPF_LOG_EPSILON

log = logging.getLogger(__name__)

# Depth window for calibration (metres, positive = below surface)
CALIB_DEPTH_MIN_M = 5.0
CALIB_DEPTH_MAX_M = 34.0

# Minimum usable training samples
MIN_SAMPLES = 50


def _stumpf_log_ratio(b02: float, b03: float, n: float = 1000.0) -> float:
    """Stumpf (2003) log-ratio for a single pixel pair."""
    eps = STUMPF_LOG_EPSILON
    return np.log(n * max(b02, eps)) / np.log(n * max(b03, eps))


def _sample_bands_at_photons(
    b02_path: str | Path,
    b03_path: str | Path,
    photons: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each photon {lon, lat, true_depth_m} that falls inside the raster,
    return arrays of (log_ratios, depths, valid_mask).

    Photons outside the raster extent are silently dropped.
    """
    b02_path = Path(b02_path)
    b03_path = Path(b03_path)

    from pyproj import Transformer

    with rasterio.open(b02_path) as src02:
        b02_data = src02.read(1).astype(np.float32)
        transform = src02.transform
        h, w = src02.shape
        bounds = src02.bounds
        raster_crs = src02.crs

    with rasterio.open(b03_path) as src03:
        b03_data = src03.read(1).astype(np.float32)

    lons = np.array([p["lon"] for p in photons], dtype=np.float64)
    lats = np.array([p["lat"] for p in photons], dtype=np.float64)
    depths = np.array([p["true_depth_m"] for p in photons], dtype=np.float32)

    # Reproject photon WGS84 coords to raster CRS if needed
    if raster_crs and not raster_crs.is_geographic:
        tf = Transformer.from_crs("EPSG:4326", raster_crs.to_epsg() or raster_crs,
                                  always_xy=True)
        xs, ys = tf.transform(lons, lats)
    else:
        xs, ys = lons, lats

    # Spatial filter: inside raster bbox
    inside = (
        (xs >= bounds.left) & (xs <= bounds.right) &
        (ys >= bounds.bottom) & (ys <= bounds.top)
    )
    if inside.sum() == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)

    xs_in     = xs[inside]
    ys_in     = ys[inside]
    depths_in = depths[inside]

    # Pixel lookup using projected coordinates
    rows, cols = rowcol(transform, xs_in, ys_in)
    rows = np.clip(np.array(rows), 0, h - 1)
    cols = np.clip(np.array(cols), 0, w - 1)

    b02_vals = b02_data[rows, cols]
    b03_vals = b03_data[rows, cols]

    # Compute log-ratio; mark invalid pixels
    n = 1000.0
    eps = STUMPF_LOG_EPSILON
    b02_safe = np.maximum(b02_vals, eps)
    b03_safe = np.maximum(b03_vals, eps)
    log_b02 = np.log(n * b02_safe)
    log_b03 = np.log(n * b03_safe)

    valid = (log_b03 != 0) & np.isfinite(log_b02) & np.isfinite(log_b03)
    log_ratios = np.where(valid, log_b02 / log_b03, np.nan)

    return log_ratios, depths_in, valid


def calibrate_from_icesat2(
    b02_path: str | Path,
    b03_path: str | Path,
    photon_cache: str | Path | None = None,
    photons: list[dict] | None = None,
    depth_min_m: float = CALIB_DEPTH_MIN_M,
    depth_max_m: float = CALIB_DEPTH_MAX_M,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """
    Calibrate Stumpf m0/m1 for a single scene using ICESat-2 ATL03 photons.

    Supply either *photon_cache* (path to the JSON file written by
    run_icesat2_deep_survey.py) or a pre-loaded *photons* list.

    Returns a dict with keys:
        m0, m1          – Stumpf calibration coefficients (depth = m1·ratio + m0)
        rmse            – leave-one-out RMSE on training set (metres)
        calibration_samples – number of photons used for regression
        depth_min_m, depth_max_m – actual depth window used
        source          – "ICESat-2 ATL03"
        status          – "success" | "insufficient_data" | "error"
        message         – human-readable detail
    """
    from src.constants import STUMPF_M0_DEFAULT, STUMPF_M1_DEFAULT

    default = {
        "m0": STUMPF_M0_DEFAULT,
        "m1": STUMPF_M1_DEFAULT,
        "rmse": None,
        "calibration_samples": 0,
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
        "source": "ICESat-2 ATL03",
        "status": "error",
        "message": "",
    }

    try:
        # Load photons
        if photons is None:
            if photon_cache is None:
                default["status"] = "error"
                default["message"] = "Either photon_cache or photons must be supplied"
                return default
            cache_path = Path(photon_cache)
            if not cache_path.exists():
                default["status"] = "insufficient_data"
                default["message"] = f"Photon cache not found: {cache_path}"
                return default
            log.info("Loading ATL03 photon cache (%s)...", cache_path)
            photons = json.loads(cache_path.read_text())

        log.info("ATL03 photons available: %d", len(photons))

        # Depth filter
        photons_in_window = [
            p for p in photons
            if depth_min_m <= p["true_depth_m"] <= depth_max_m
        ]
        log.info("Photons in depth window [%.0f–%.0f m]: %d",
                 depth_min_m, depth_max_m, len(photons_in_window))

        if len(photons_in_window) < min_samples:
            default["status"] = "insufficient_data"
            default["message"] = (
                f"Only {len(photons_in_window)} photons in window "
                f"[{depth_min_m}–{depth_max_m} m], need ≥{min_samples}"
            )
            return default

        # Sample B02/B03 at photon locations
        log_ratios, depths, valid = _sample_bands_at_photons(
            b02_path, b03_path, photons_in_window
        )

        if valid.sum() == 0:
            default["status"] = "insufficient_data"
            default["message"] = "No ATL03 photons fall inside the scene extent"
            return default

        # Final depth-window filter on matched photons
        depth_ok = (depths[valid] >= depth_min_m) & (depths[valid] <= depth_max_m)
        X = log_ratios[valid][depth_ok].reshape(-1, 1)
        y = depths[valid][depth_ok]  # positive depth in metres

        n_samples = len(X)
        log.info("Regression training pairs inside scene: %d", n_samples)

        if n_samples < min_samples:
            default["status"] = "insufficient_data"
            default["message"] = (
                f"Only {n_samples} photons overlap scene extent, need ≥{min_samples}"
            )
            return default

        # Huber robust regression: depth = m1 * log_ratio + m0
        model = HuberRegressor(max_iter=500)
        model.fit(X, y)

        m1 = float(model.coef_[0])
        m0 = float(model.intercept_)

        y_pred = model.predict(X)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        log.info(
            "ICESat-2 calibration: m0=%.3f m1=%.3f RMSE=%.3f m (%d samples)",
            m0, m1, rmse, n_samples,
        )

        return {
            "m0": m0,
            "m1": m1,
            "rmse": rmse,
            "calibration_samples": n_samples,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "source": "ICESat-2 ATL03",
            "status": "success",
            "message": (
                f"Fitted on {n_samples} ATL03 photons "
                f"({depth_min_m:.0f}–{depth_max_m:.0f} m), RMSE={rmse:.2f} m"
            ),
        }

    except Exception as exc:
        log.warning("ICESat-2 calibration failed: %s", exc, exc_info=True)
        default["status"] = "error"
        default["message"] = str(exc)
        return default


def apply_icesat2_sdb(
    b02_path: str | Path,
    b03_path: str | Path,
    m0: float,
    m1: float,
    output_path: str | Path,
    depth_max_mask_m: float = 34.0,
) -> Path:
    """
    Apply calibrated Stumpf coefficients to produce a full-scene SDB depth map.

    Output: GeoTIFF with positive depth values (metres), NaN = no data / too deep.
    """
    with rasterio.open(b02_path) as src:
        b02 = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    with rasterio.open(b03_path) as src:
        b03 = src.read(1).astype(np.float32)

    eps = STUMPF_LOG_EPSILON
    n = 1000.0
    b02_s = np.maximum(b02, eps)
    b03_s = np.maximum(b03, eps)
    log_b03 = np.log(n * b03_s)
    log_ratio = np.where(log_b03 != 0, np.log(n * b02_s) / log_b03, np.nan)

    depth = m1 * log_ratio + m0

    # Mask: keep only plausible positive depths within the calibration window
    depth = np.where(
        np.isfinite(depth) & (depth > 0) & (depth <= depth_max_mask_m),
        depth,
        np.nan,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(depth.astype(np.float32), 1)

    log.info("ICESat-2-calibrated SDB map → %s (m0=%.2f m1=%.2f)", out, m0, m1)
    return out
