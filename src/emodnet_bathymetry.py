"""
emodnet_bathymetry.py — EMODnet Bathymetry WCS ingestion and rugosity computation.

Downloads mean depth from the EMODnet Bathymetry WCS service (v1.0.0) for a given
bounding box, then derives:
  - Terrain Ruggedness Index (TRI): mean absolute difference between each cell and
    its 8 neighbours — high TRI ≈ structural complexity ≈ rocky reef.
  - A binary habitat mask: depth in optical window [0, -max_depth_m] AND TRI above
    a reef-threshold → reef; flat seafloor below threshold → sand candidate.

Resolution: EMODnet mean grid is 1/16 arc-minute (≈ 115 m at 37 °N).  This is
sufficient to discriminate reef *zones* from flat sandy shelf at patch scale
(128 × 10 m Sentinel-2 pixels = 1.28 km); it cannot resolve individual
coral heads (<10 m).

WCS endpoint:  https://ows.emodnet-bathymetry.eu/wcs
Coverage name: emodnet:mean  (negative = below sea level)

Usage
-----
    from src.emodnet_bathymetry import fetch_depth, compute_tri, build_habitat_mask

    depth_path = fetch_depth(
        bbox=(-8.6, 36.9, -7.7, 37.3),
        output_path="outputs/reef_habitat/emodnet/depth_algarve.tif",
    )
    tri_path = compute_tri(depth_path, "outputs/reef_habitat/emodnet/tri_algarve.tif")
    mask      = build_habitat_mask(depth_path, tri_path)
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_bounds

log = logging.getLogger(__name__)

# ── WCS constants ─────────────────────────────────────────────────────────────
_WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
_COVERAGE = "emodnet:mean"
_WCS_VERSION = "1.0.0"
_NATIVE_RES = 1 / 16 / 60  # 1/16 arc-minute in decimal degrees ≈ 0.001042°
_TIMEOUT_S = 60

# ── Physical constants ─────────────────────────────────────────────────────────
DEFAULT_MAX_DEPTH_M = 25.0  # Sentinel-2 optical limit in clear Algarve water
DEFAULT_TRI_REEF = 0.30  # TRI (m) above which a cell is classified as reef-like
# Tuned for EMODnet 115 m grid over Algarve shelf


# ─────────────────────────────────────────────────────────────────────────────
# 1. Depth download
# ─────────────────────────────────────────────────────────────────────────────


def fetch_depth(
    bbox: tuple[float, float, float, float],
    output_path: str | Path,
    *,
    resx: float = _NATIVE_RES,
    resy: float = _NATIVE_RES,
    timeout: int = _TIMEOUT_S,
) -> Path:
    """
    Download EMODnet mean depth for *bbox* and write a GeoTIFF to *output_path*.

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max) in WGS-84 decimal degrees.
    output_path : Destination path for the GeoTIFF.
    resx, resy : Pixel size in degrees (default: native 1/16 arc-minute).
    timeout : HTTP timeout in seconds.

    Returns
    -------
    Path to the written GeoTIFF.

    Raises
    ------
    RuntimeError : If the WCS request fails or returns no data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    width = max(1, round((lon_max - lon_min) / resx))
    height = max(1, round((lat_max - lat_min) / resy))

    params = {
        "service": "WCS",
        "version": _WCS_VERSION,
        "request": "GetCoverage",
        "coverage": _COVERAGE,
        "crs": "EPSG:4326",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "width": str(width),
        "height": str(height),
        "format": "GeoTIFF",
    }
    log.info(
        "Fetching EMODnet depth  bbox=%s  grid=%d×%d  → %s",
        bbox,
        width,
        height,
        output_path,
    )

    resp = requests.get(_WCS_URL, params=params, timeout=timeout, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"EMODnet WCS returned HTTP {resp.status_code}: {resp.text[:200]}")

    # Check Content-Type; WCS error responses are XML even on 200
    ct = resp.headers.get("Content-Type", "")
    if "xml" in ct.lower() or "text" in ct.lower():
        raise RuntimeError(f"EMODnet WCS returned non-raster content: {resp.text[:400]}")

    raw = resp.content
    if len(raw) < 1_000:
        raise RuntimeError(f"EMODnet WCS response too small ({len(raw)} bytes) — likely an error")

    # Parse in-memory and re-write with correct metadata
    with rasterio.open(BytesIO(raw)) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()

    # Ensure correct georeferencing (WCS sometimes omits transform)
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    profile.update(
        driver="GTiff",
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
        width=width,
        height=height,
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    # Normalise nodata
    if nodata is not None:
        data[data == nodata] = -9999.0
    # EMODnet uses positive integers for depth above MSL (land); keep only marine
    # negative values (below sea level) and NaN everything else
    data[data > 1.0] = -9999.0  # land / positive elevation → nodata

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)

    valid_px = int((data != -9999.0).sum())
    log.info(
        "EMODnet depth saved: %s  (%d × %d px, %d valid, %.1f–%.1f m)",
        output_path,
        height,
        width,
        valid_px,
        float(data[data != -9999.0].min()) if valid_px else 0,
        float(data[data != -9999.0].max()) if valid_px else 0,
    )
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Terrain Ruggedness Index
# ─────────────────────────────────────────────────────────────────────────────


def compute_tri(
    depth_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Compute Terrain Ruggedness Index (TRI) from a depth GeoTIFF.

    TRI(i,j) = mean( |depth(i,j) - depth(k,l)| )  for (k,l) in 3×3 neighbourhood.

    High TRI → heterogeneous bathymetry → structural complexity → reef indicator.
    Low TRI  → flat seafloor → sandy shelf.

    Parameters
    ----------
    depth_path  : Path to depth GeoTIFF (negative values = below sea level).
    output_path : Destination GeoTIFF for TRI values.

    Returns
    -------
    Path to the written TRI GeoTIFF.
    """
    depth_path = Path(depth_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(depth_path) as src:
        depth = src.read(1).astype(np.float64)
        nodata = src.nodata or -9999.0
        profile = src.profile.copy()

    valid = depth != nodata
    depth_ma = np.where(valid, depth, np.nan)

    # Pad with NaN for border handling
    padded = np.pad(depth_ma, 1, constant_values=np.nan)
    tri = np.full_like(depth_ma, np.nan)

    # 3×3 neighbourhood — unrolled for speed
    rows, cols = depth_ma.shape
    centre = padded[1 : rows + 1, 1 : cols + 1]
    diffs = np.zeros_like(centre)
    count = np.zeros_like(centre)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nbr = padded[1 + dr : rows + 1 + dr, 1 + dc : cols + 1 + dc]
            mask = ~np.isnan(centre) & ~np.isnan(nbr)
            diffs = np.where(mask, diffs + np.abs(centre - nbr), diffs)
            count = np.where(mask, count + 1, count)

    with np.errstate(invalid="ignore", divide="ignore"):
        tri = np.where(count > 0, diffs / count, np.nan)

    profile.update(
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    tri_out = np.where(np.isnan(tri), -9999.0, tri).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(tri_out, 1)

    valid_tri = tri_out[tri_out != -9999.0]
    log.info(
        "TRI saved: %s  (valid=%d  p50=%.3f m  p95=%.3f m  max=%.3f m)",
        output_path,
        len(valid_tri),
        np.percentile(valid_tri, 50) if valid_tri.size else 0,
        np.percentile(valid_tri, 95) if valid_tri.size else 0,
        valid_tri.max() if valid_tri.size else 0,
    )
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Habitat mask
# ─────────────────────────────────────────────────────────────────────────────


def build_habitat_mask(
    depth_path: str | Path,
    tri_path: str | Path,
    *,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
    tri_reef_threshold: float = DEFAULT_TRI_REEF,
) -> dict:
    """
    Combine depth and TRI to produce reef / sand candidate classification.

    Classes
    -------
    2  reef_candidate   : within optical depth window AND TRI ≥ threshold
    1  sand_candidate   : within optical depth window AND TRI < threshold
    0  outside_window   : depth > max_depth_m (too deep for Sentinel-2)
    -1 nodata           : missing depth or TRI

    Parameters
    ----------
    depth_path          : EMODnet depth GeoTIFF (negative = below MSL).
    tri_path            : TRI GeoTIFF from compute_tri().
    max_depth_m         : Optical depth limit in metres (default 25 m).
    tri_reef_threshold  : TRI (m) above which a cell is reef-like (default 0.30).

    Returns
    -------
    dict with keys:
        mask        : int8 ndarray (classes as above)
        reef_frac   : fraction of optical-window pixels classified as reef
        sand_frac   : fraction classified as sand
        n_reef_px   : count of reef pixels
        n_sand_px   : count of sand pixels
        transform   : rasterio Affine transform
        crs         : rasterio CRS
    """
    with rasterio.open(depth_path) as src:
        depth = src.read(1).astype(np.float32)
        nd_d = src.nodata or -9999.0
        transform = src.transform
        crs = src.crs

    with rasterio.open(tri_path) as src:
        tri = src.read(1).astype(np.float32)
        nd_t = src.nodata or -9999.0

    mask = np.full(depth.shape, -1, dtype=np.int8)

    # Valid pixels: both layers have data
    valid = (depth != nd_d) & (tri != nd_t)

    # Depth in optical window: 0 to -max_depth_m (EMODnet uses negative for depth)
    in_window = valid & (depth >= -max_depth_m) & (depth <= 0.0)

    mask[in_window & (tri >= tri_reef_threshold)] = 2  # reef candidate
    mask[in_window & (tri < tri_reef_threshold)] = 1  # sand candidate
    mask[valid & ~in_window] = 0  # valid but too deep / above sea level

    n_reef = int((mask == 2).sum())
    n_sand = int((mask == 1).sum())
    n_win = n_reef + n_sand

    log.info(
        "Habitat mask: reef=%d px (%.1f%%)  sand=%d px (%.1f%%)  " "outside_window=%d px  nodata=%d px",
        n_reef,
        100 * n_reef / n_win if n_win else 0,
        n_sand,
        100 * n_sand / n_win if n_win else 0,
        int((mask == 0).sum()),
        int((mask == -1).sum()),
    )

    return {
        "mask": mask,
        "reef_frac": n_reef / n_win if n_win else 0.0,
        "sand_frac": n_sand / n_win if n_win else 0.0,
        "n_reef_px": n_reef,
        "n_sand_px": n_sand,
        "transform": transform,
        "crs": crs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Convenience: sample a single point
# ─────────────────────────────────────────────────────────────────────────────


def sample_depth_at_point(
    depth_path: str | Path,
    lon: float,
    lat: float,
) -> float | None:
    """
    Return the EMODnet depth value (metres, negative=below MSL) at (lon, lat).
    Returns None if the point is outside the raster or has nodata.
    """
    from rasterio.transform import rowcol

    with rasterio.open(depth_path) as src:
        nodata = src.nodata or -9999.0
        try:
            row, col = rowcol(src.transform, lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = float(src.read(1)[row, col])
                return None if val == nodata else val
        except Exception:
            pass
    return None
