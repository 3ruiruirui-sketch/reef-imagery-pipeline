#!/usr/bin/env python3
"""
Utilities: raster I/O, sunglint correction, refraction, Beer-Lambert.
"""
import math
import numpy as np
import rasterio

def read_band(path, handle_nodata=True):
    """Read raster band as float32. If handle_nodata=True, converts nodata to np.nan."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if handle_nodata and nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile

def write_band(path, arr, profile, nodata=None):
    """Write array to GeoTIFF. If nodata is None and arr contains NaN, uses np.nan as nodata."""
    profile = profile.copy()
    has_nan = np.isnan(arr).any()
    if nodata is None and has_nan:
        nodata = np.nan
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=nodata)
    with rasterio.open(str(path), 'w', **profile) as dst:
        dst.write(arr.astype(np.float32), 1)

def simulate_acolite_boa(input_tif, output_tif, b03_tif=None, sunglint_strength=0.8):
    """
    Simulates ACOLITE BOA from L2A B02:
    - Hedley linear sunglint correction (if B03 provided), else empirical subtraction.
    - Converts raw DN to BOA reflectance (divide by 10000).
    - No negative values.
    Drop-in: replace with real acolite_cli output GeoTIFFs and remove this function.
    """
    arr, profile = read_band(str(input_tif))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert L2A DN to reflectance if not already (values > 2 = raw DN)
    if arr.max() > 2.0:
        arr = arr / 10000.0

    if b03_tif is not None:
        b03, _ = read_band(str(b03_tif))
        if b03.max() > 2.0:
            b03 = b03 / 10000.0
        # Hedley linear: slope from deep-water pixels
        mask = (arr > 0) & (b03 > 0) & (b03 < np.percentile(b03[b03 > 0], 20))
        if mask.sum() > 10:
            slope = np.cov(arr[mask].ravel(), b03[mask].ravel())[0, 1] / (np.var(b03[mask]) + 1e-12)
            slope = np.clip(slope, 0, 2.0)
            arr = arr - slope * (b03 - b03[mask].min())
    else:
        # Empirical: subtract fraction of high-percentile tail
        p95 = np.percentile(arr[arr > 0], 95) if np.any(arr > 0) else 0.0
        arr = arr - sunglint_strength * p95 * 0.05

    arr = np.clip(arr, 0, 1.0)  # physical reflectance range; NO min-max stretch
    write_band(str(output_tif), arr.astype(np.float32), profile)
    return str(output_tif)

def snell_air_to_water(theta_air_rad, n_water=1.333):
    """Snell's law: returns refracted angle (radians) in water for ray entering from air."""
    s = math.sin(theta_air_rad) / n_water
    s = max(-0.999999, min(0.999999, s))
    return math.asin(s)  # angle in [0, pi/2]

def snell_sza(sza_deg, n_water=1.333):
    """Return (sza_water_deg, theta_water_rad)."""
    rad = math.radians(sza_deg)
    sin_w = math.sin(rad) / n_water
    sin_w = max(-0.999999, min(0.999999, sin_w))
    theta_w = math.asin(sin_w)
    return math.degrees(theta_w), theta_w

def optical_path(depth_m, theta_water_rad):
    return depth_m / max(1e-6, math.cos(theta_water_rad))

def beer_lambert_transmittance(kd, path_m):
    """Two-way transmittance (surface → bottom → sensor)."""
    return math.exp(-2 * kd * path_m)

def get_kd490(month, kd_prior: dict):
    return kd_prior.get(str(month).zfill(2), kd_prior.get(str(month), 0.080))

def compute_metadata_stub(date):
    """
    Minimal metadata for simulated mode.
    Replace with real STAC/MTD metadata extraction in production.
    """
    known = {
        "2025-09-25": {"sza": 40.498, "saa": 158.883, "cloud": 1.245, "level": "L2A"},
        "2023-10-01": {"sza": 42.413, "saa": 160.459, "cloud": 0.007, "level": "L2A"},
    }
    m = known.get(date, {"sza": 40.0, "saa": 150.0, "cloud": 2.0, "level": "L2A"})
    return {
        "date": date,
        "crs": "EPSG:32629",
        "datum": "WGS84",
        "level": m["level"],
        "solar_zenith_deg": m["sza"],
        "solar_azimuth_deg": m["saa"],
        "satellite_zenith_deg": 5.0,
        "satellite_azimuth_deg": 10.0,
        "cloud_cover_pct": m["cloud"],
    }
