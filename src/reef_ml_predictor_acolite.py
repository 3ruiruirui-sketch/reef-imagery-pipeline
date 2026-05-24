#!/usr/bin/env python3
"""
reef_ml_predictor_acolite.py — v3.0
Adds:
  (A) Full Gordon/QAA-style Kd inversion using B02/B03/B04 bands
  (B) Stumpf log-ratio Satellite Derived Bathymetry (SDB) depth map
  (C) Gordon band-ratio Kd estimator integrated into run_predictor
"""
import math, logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import read_band, write_band, snell_sza, optical_path, beer_lambert_transmittance, get_kd490
try:
    from src.bathy_calibrator import run_bathy_integration
    _BATHY_AVAILABLE = True
except ImportError:
    _BATHY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Physical constants ────────────────────────────────────────────────────────
from src.constants import (
    DEFAULT_DEPTH_TARGET, SDB_OPTICAL_LIMIT_M,
    STUMPF_M0_DEFAULT, STUMPF_M1_DEFAULT, STUMPF_M1_LITERATURE, STUMPF_N,
    KD490_TABLE, KD490_DEFAULT, GLINT_PENALTY, GLINT_PENALTY_DEFAULT,
    N_WATER, SNR_THRESHOLD, BUF_PIX, SAND_R, ROCK_R,
    REFLECTANCE_DN_SCALE, REFLECTANCE_DN_THRESHOLD,
)

# Aliases for backward compat
DEPTH_TARGET    = DEFAULT_DEPTH_TARGET
DEFAULT_KD_TABLE = KD490_TABLE

# Sentinel-2 band pure-water attenuation (aw, m⁻¹) — Pope & Fry 1997
AW = {"B02": 0.0145, "B03": 0.0612, "B04": 0.4300}

# ── A: Gordon/QAA Kd inversion ───────────────────────────────────────────────
def gordon_kd_inversion(b02: np.ndarray, b03: np.ndarray,
                         b04: np.ndarray | None = None,
                         kd_prior: float = 0.045) -> tuple[float, float, float]:
    """
    Quasi-Analytical Algorithm (QAA, Lee et al. 2002) simplified for B02/B03/B04.
    Returns: (kd_b02_est, kd_b03_est, kd_b04_est)

    Step 1 – rrs (sub-surface remote sensing reflectance):
        rrs = Rrs / (0.52 + 1.7 * Rrs)   [Lee 2002, eq. 4]

    Step 2 – u (ratio bb / (a + bb)):
        u = (-g0 + sqrt(g0² + 4*g1*rrs)) / (2*g1)
        g0=0.0895, g1=0.1247  [Gordon 1988]

    Step 3 – total absorption at 555nm (green):
        a555 = aw_555 + 10^(-1.146 - 1.366*chi - 0.469*chi²)   where chi = log10(rrs_B02/rrs_B03)

    Step 4 – Kd at each band:
        Kd = (1 + 0.005 * sza) * a + 4.18*(1 - 0.52*exp(-10.8*a)) * bb
        Simplified: Kd ≈ (a + bb) / cos(theta_sun)   [Morel 2007]
    """
    G0, G1 = 0.0895, 0.1247
    AW_B02, AW_B03 = AW["B02"], AW["B03"]
    AW_B04         = AW["B04"]

    def to_rrs(rrs_surf):
        return rrs_surf / (0.52 + 1.7 * rrs_surf + 1e-9)

    mask = (b02 > 0) & (b03 > 0)
    if mask.sum() < 10:
        return kd_prior, kd_prior * (490/560)**0.5, kd_prior * 0.1

    rrs02 = to_rrs(b02[mask])
    rrs03 = to_rrs(b03[mask])

    # Step 2: u at green (B03)
    u_green = (-G0 + np.sqrt(G0**2 + 4 * G1 * rrs03 + 1e-12)) / (2 * G1)
    u_blue  = (-G0 + np.sqrt(G0**2 + 4 * G1 * rrs02 + 1e-12)) / (2 * G1)

    # Step 3: total absorption at green (560nm)
    chi = np.log10(np.clip(rrs02 / (rrs03 + 1e-9), 1e-3, 100))
    a555 = AW_B03 + 10 ** (-1.146 - 1.366 * chi - 0.469 * chi**2)

    # Step 4: backscattering at green
    bb555 = u_green * a555 / (1 - u_green + 1e-9)
    bb555 = np.clip(bb555, 0, 1)

    # Scale bb to blue (spectral power law: bb(λ) ~ bb(555) * (555/λ)^Y, Y≈1)
    bb_blue = bb555 * (560 / 490)

    # Total absorption at blue
    a_blue = AW_B02 + bb_blue * (1 - u_blue + 1e-9) / (u_blue + 1e-9)
    a_blue = np.clip(a_blue, AW_B02, 1.0)

    # Kd ≈ a + bb  (nadir viewing, simplified Morel)
    kd_b02 = float(np.nanmedian(a_blue + bb_blue))
    kd_b03 = float(np.nanmedian(a555   + bb555))

    # B04 (665nm) — use simple Gordon power-law scaling from B03
    kd_b04 = kd_b03 * (AW_B04 / AW_B03 * 0.8) if b04 is None else _kd_from_band(b04, AW_B04)

    # Sanity clamp: Algarve coastal range
    kd_b02 = float(np.clip(kd_b02, 0.010, 0.500))
    kd_b03 = float(np.clip(kd_b03, 0.020, 0.500))
    kd_b04 = float(np.clip(kd_b04, 0.050, 2.000))

    return kd_b02, kd_b03, kd_b04

def _kd_from_band(band: np.ndarray, aw: float) -> float:
    """Fallback: estimate Kd from single band reflectance level."""
    mask = band > 0
    if mask.sum() < 5:
        return aw * 3
    rrs = band[mask] / (0.52 + 1.7 * band[mask])
    G0, G1 = 0.0895, 0.1247
    u = (-G0 + np.sqrt(G0**2 + 4 * G1 * rrs + 1e-12)) / (2 * G1)
    a = aw + u * aw / (1 - u + 1e-9) * 0.5
    bb = np.clip(u * a / (1 - u + 1e-9), 0, 1)
    return float(np.nanmedian(a + bb))

# ── B: Stumpf Log-Ratio SDB depth map ────────────────────────────────────────
def stumpf_sdb(b02: np.ndarray, b03: np.ndarray,
               m0: float = STUMPF_M0_DEFAULT,
               m1: float = STUMPF_M1_DEFAULT,
               n: float = STUMPF_N) -> np.ndarray:
    """
    Stumpf et al. (2003) log-ratio Satellite Derived Bathymetry:
        depth = m1 * ln(n * B02) / ln(n * B03) + m0
    Default m0/m1 calibrated for Algarve oligotrophic waters (Kd≈0.045).
    Returns depth map in metres (positive = deeper). Values >40m set to NaN
    (optical limit exceeded, unreliable extrapolation).
    """
    eps = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            (b02 > eps) & (b03 > eps),
            np.log(n * b02 + eps) / (np.log(n * b03 + eps) + eps),
            np.nan
        )
    depth = m1 * ratio + m0
    # Clip negative values to 0, but set beyond optical limit to NaN
    depth = np.where(depth < 0, 0, depth)
    depth = np.where(depth > SDB_OPTICAL_LIMIT_M, np.nan, depth)
    return depth.astype(np.float32)

# ── C: Integrated Kd estimator (simple band-ratio, fallback when QAA fails) ──
def estimate_kd_bandratio(b02: np.ndarray, b03: np.ndarray,
                          kd_prior: float) -> tuple[float, bool]:
    """Gordon approximation: Kd scales with B02/B03 ratio residual."""
    mask = (b02 > 0) & (b03 > 0)
    if mask.sum() < 10:
        return kd_prior, False
    ratio = np.mean(b02[mask]) / (np.mean(b03[mask]) + 1e-9)
    kd_est = kd_prior * (1 + (ratio - 1.0) * 0.15)
    high_uncert = abs(kd_est - kd_prior) / kd_prior > 0.30
    return float(np.clip(kd_est, 0.010, 0.500)), high_uncert

# ── SNR map ───────────────────────────────────────────────────────────────────
def make_snr_map(arr: np.ndarray, window: int = 7) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter
        m  = uniform_filter(arr.astype(np.float64), size=window)
        sq = uniform_filter(arr.astype(np.float64)**2, size=window)
        std = np.sqrt(np.clip(sq - m**2, 0, None))
        return np.where(std > 0, m / std, 0).astype(np.float32)
    except ImportError:
        sig = np.mean(arr[arr > 0]) if np.any(arr > 0) else 0
        std = np.std(arr[arr > 0]) + 1e-9
        return np.full_like(arr, sig / std, dtype=np.float32)

# ── Main predictor ────────────────────────────────────────────────────────────
def run_predictor(boa_b02_path, metadata, output_dir,
                  kd_prior: dict | None = None, cloud_threshold: float = 0.2,
                  snr_threshold: float = 3.0, date: str | None = None,
                  b03_path: str | None = None, b04_path: str | None = None,
                  lat: float | None = None, lon: float | None = None,
                  depth_target: float = DEFAULT_DEPTH_TARGET) -> dict:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date    = date or metadata.get("date", "unknown")
    month   = int(date.split("-")[1]) if "-" in date else 9
    kd_tbl  = kd_prior or DEFAULT_KD_TABLE
    kd_seas = get_kd490(month, kd_tbl)
    glint_pen = GLINT_PENALTY.get(month, 0.80)

    b02_arr, profile = read_band(boa_b02_path)
    b02_arr = np.nan_to_num(b02_arr, nan=0.0)
    # Normalise to BOA reflectance [0..1] — handles both DN (>2) and already-BOA inputs
    if b02_arr.max() > 2.0:
        b02_arr = b02_arr / 10000.0

    b03_arr = b04_arr = None
    if b03_path:
        b03_arr, _ = read_band(b03_path)
        if b03_arr.max() > 2.0: b03_arr /= 10000.0
    if b04_path:
        b04_arr, _ = read_band(b04_path)
        if b04_arr.max() > 2.0: b04_arr /= 10000.0

    # ── Kd estimation: QAA if B03 available, else band-ratio, else prior ──────
    kd_method = "seasonal_prior"
    kd_high_uncert = False
    kd_b02, kd_b03, kd_b04 = kd_seas, kd_seas, kd_seas

    if b03_arr is not None:
        try:
            kd_b02, kd_b03, kd_b04 = gordon_kd_inversion(b02_arr, b03_arr, b04_arr, kd_seas)
            # If QAA saturated to boundary, fall back to band-ratio
            if kd_b02 >= 0.499:
                raise ValueError("QAA Kd saturated — using band-ratio fallback")
            kd_high_uncert = abs(kd_b02 - kd_seas) / kd_seas > 0.30
            kd_method = "gordon_qaa"
            logging.info("Gordon/QAA Kd: B02=%.4f B03=%.4f B04=%.4f (prior=%.4f)",
                         kd_b02, kd_b03, kd_b04, kd_seas)
        except Exception as e:
            logging.warning("Gordon inversion issue (%s) — falling back to band-ratio", e)
            kd_b02, kd_high_uncert = estimate_kd_bandratio(b02_arr, b03_arr, kd_seas)
            kd_b03 = kd_b02 * (490 / 560)**0.5
            kd_method = "band_ratio_fallback"
    else:
        logging.info("No B03 — using seasonal Kd prior %.4f", kd_seas)

    if kd_high_uncert:
        logging.warning("Kd diverges >30%% from prior — high uncertainty flag")

    # ── Physics: Snell + Beer-Lambert ─────────────────────────────────────────
    sza_deg = metadata.get("solar_zenith_deg", 40.5)
    sza_water_deg, theta_water = snell_sza(sza_deg)
    path_m = optical_path(depth_target, theta_water)
    trans  = beer_lambert_transmittance(kd_b02, path_m)

    # ── Bottom reflectance estimate ───────────────────────────────────────────
    bottom_est = np.clip(b02_arr * math.exp(-kd_b02 * path_m), 0, None).astype(np.float32)

    # ── SNR map ───────────────────────────────────────────────────────────────
    snr_map    = make_snr_map(bottom_est)
    snr_mean   = float(np.nanmean(snr_map))
    snr_median = float(np.nanmedian(snr_map))

    # ── SDB depth map (Stumpf) — with IH chart calibration ───────────────────
    sdb_map = None
    bathy_result = {}
    stumpf_m0 = -16.0
    stumpf_m1 = 20.0

    if b03_arr is not None:
        # Try to calibrate Stumpf coefficients from IH isobaths
        calibration_status = "skipped_no_location"  # default if lat/lon missing
        if _BATHY_AVAILABLE and lat is not None and lon is not None:
            calibration_status = "skipped_no_transform"  # will be overwritten if transform exists
            try:
                tf = profile.get("transform")
                if tf is not None:
                    h, w = b02_arr.shape
                    min_lon_r = tf.c
                    max_lat_r = tf.f
                    max_lon_r = tf.c + tf.a * w
                    min_lat_r = tf.f + tf.e * h
                    bounds_wgs = (min_lat_r, min_lon_r, max_lat_r, max_lon_r)
                    bathy_result = run_bathy_integration(
                        lat=lat, lon=lon,
                        b02_arr=b02_arr, b03_arr=b03_arr,
                        bounds_wgs84=bounds_wgs,
                    )
                    stumpf_m0 = bathy_result.get("recommended_m0", -16.0)
                    stumpf_m1 = bathy_result.get("recommended_m1", 20.0)
                    zone_info = bathy_result.get("zone", {})
                    calibrated = bathy_result.get("calibration", {}).get("calibrated", False)
                    calibration_status = "success" if calibrated else "failed_insufficient_data"
                    logging.info(
                        "IH bathy zone=%s | optically_viable=%s | "
                        "Stumpf m0=%.2f m1=%.2f (calibrated=%s, status=%s)",
                        zone_info.get("zone"), zone_info.get("optically_viable"),
                        stumpf_m0, stumpf_m1, calibrated, calibration_status
                    )
                else:
                    logging.warning("IH calibration skipped: no raster transform in profile")
            except Exception as bathy_err:
                calibration_status = f"failed_error: {type(bathy_err).__name__}"
                logging.warning("IH calibration failed with error: %s — using Stumpf defaults", bathy_err)
        else:
            if not _BATHY_AVAILABLE:
                calibration_status = "skipped_module_unavailable"
                logging.info("IH calibration skipped: bathy_calibrator module not available")
            elif lat is None or lon is None:
                calibration_status = "skipped_no_location"
                logging.info("IH calibration skipped: lat/lon not provided")
        bathy_result["calibration_status"] = calibration_status

        # Compute SDB with (possibly calibrated) coefficients
        sdb_map = stumpf_sdb(b02_arr, b03_arr, m0=stumpf_m0, m1=stumpf_m1)
        sdb_path = out / "sdb_depth_map.tif"
        write_band(str(sdb_path), sdb_map, profile)
        sdb_mean = float(np.nanmean(sdb_map[sdb_map > 0]))
        logging.info("SDB depth map: mean=%.1fm, written to %s", sdb_mean, sdb_path)

        # Validate SDB vs IH chart (if calibration ran)
        if bathy_result and _BATHY_AVAILABLE and lat is not None:
            try:
                from src.bathy_calibrator import validate_sdb_vs_chart, fetch_isobaths_for_bbox
                deg_buf = 3000 / 111_000.0
                feats = fetch_isobaths_for_bbox(
                    lon - deg_buf, lat - deg_buf, lon + deg_buf, lat + deg_buf
                )
                if tf is not None:
                    val = validate_sdb_vs_chart(sdb_map, feats, bounds_wgs)
                    bathy_result["validation"] = val
                    ov = val.get("overall", {})
                    if ov:
                        logging.info(
                            "SDB validation vs IH chart: bias=%.2fm RMSE=%.2fm n=%d",
                            ov.get("overall_bias_m", 0),
                            ov.get("overall_rmse_m", 0),
                            ov.get("n_total", 0)
                        )
            except Exception as val_err:
                logging.warning("SDB validation failed: %s", val_err)
    else:
        sdb_path, sdb_mean = None, None

    # ── Masks & scores ────────────────────────────────────────────────────────
    cloud_pct   = metadata.get("cloud_cover_pct", 2.0)
    usable_frac = max(0.0, 1.0 - cloud_pct / 100.0)
    useful_mask = (snr_map >= snr_threshold) & (bottom_est > 0)
    pct_useful  = 100.0 * float(useful_mask.sum()) / max(1, (bottom_est > 0).sum())

    conf_map = np.where(snr_map >= snr_threshold * 2, 2,
               np.where(useful_mask, 1, 0)).astype(np.uint8)
    pct_high_conf = 100.0 * float((conf_map == 2).sum()) / max(1, (bottom_est > 0).sum())

    sand_btm = SAND_R * trans
    rock_btm = ROCK_R * trans
    contrast  = (sand_btm - rock_btm) / sand_btm if sand_btm > 0 else 0.0

    snr_ok    = min(1.0, snr_mean / 100.0)
    vis_score = min(1.0, usable_frac * snr_ok * glint_pen * contrast * 5.0)
    if kd_high_uncert:
        vis_score *= 0.80

    # ── Save GeoTIFFs ─────────────────────────────────────────────────────────
    write_band(str(out / "snr_map.tif"),         snr_map,                          profile)
    write_band(str(out / "confidence_map.tif"),  conf_map.astype(np.float32),      profile)
    write_band(str(out / "bottom_est.tif"),      bottom_est,                        profile)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary = {
        "image_date": date,
        "kd_estimation_method": kd_method,
        "kd_seasonal_prior": kd_seas,
        "kd_b02_estimated": round(kd_b02, 5),
        "kd_b03_estimated": round(kd_b03, 5),
        "kd_b04_estimated": round(kd_b04, 5),
        "kd_high_uncertainty": kd_high_uncert,
        "sza_air_deg": sza_deg,
        "sza_water_deg": round(sza_water_deg, 3),
        "optical_path_m": round(path_m, 3),
        "water_transmittance_twoway": round(trans, 5),
        "glint_penalty": glint_pen,
        "snr_mean_16m": round(snr_mean, 4),
        "snr_median_16m": round(snr_median, 4),
        "percent_pixels_useful": round(pct_useful, 2),
        "percent_area_high_confidence": round(pct_high_conf, 2),
        "contrast_benthic_mean": round(contrast, 5),
        "visibility_score": round(vis_score, 5),
        "sdb_depth_mean_m": round(sdb_mean, 2) if sdb_mean else None,
        "snr_map": str(out / "snr_map.tif"),
        "confidence_map": str(out / "confidence_map.tif"),
        "bottom_est_map": str(out / "bottom_est.tif"),
        "sdb_depth_map": str(sdb_path) if sdb_path else None,
        "stumpf_m0_used": stumpf_m0,
        "stumpf_m1_used": stumpf_m1,
        "bathy_zone":     bathy_result.get("zone", {}).get("zone") if bathy_result else None,
        "bathy_optically_viable": bathy_result.get("zone", {}).get("optically_viable") if bathy_result else None,
        "bathy_calibration_rmse_m": bathy_result.get("calibration", {}).get("rmse_m") if bathy_result else None,
        "sdb_vs_chart_bias_m": bathy_result.get("validation", {}).get("overall", {}).get("overall_bias_m") if bathy_result else None,
        "sdb_vs_chart_rmse_m": bathy_result.get("validation", {}).get("overall", {}).get("overall_rmse_m") if bathy_result else None,
    }
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    logging.info("Done | date=%s | Kd=%s(%.4f) | vis=%.4f | SNR=%.2f | SDB_mean=%.1fm",
                 date, kd_method, kd_b02, vis_score, snr_mean, sdb_mean or 0)
    return summary

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reef ML Predictor v3 — Gordon/QAA + SDB")
    p.add_argument("--boa-b02",  required=True)
    p.add_argument("--b03",      default=None)
    p.add_argument("--b04",      default=None)
    p.add_argument("--date",     required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--depth",    type=float, default=16.0)
    p.add_argument("--snr-threshold", type=float, default=3.0)
    args = p.parse_args()
    from src.utils import compute_metadata_stub
    run_predictor(args.boa_b02, compute_metadata_stub(args.date), args.output,
                  date=args.date, b03_path=args.b03, b04_path=args.b04,
                  snr_threshold=args.snr_threshold, depth_target=args.depth)
