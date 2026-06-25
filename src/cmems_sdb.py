"""
CMEMS Coastal Satellite-Derived Bathymetry — fetch and reproject.

Product: BATHYMETRY_GLO_PHY_COASTAL_L4_MY_016_001
Methods available:
  composite  →  cmems_obs-sdb_glo_phy_comp_my_100m-l4-s2_static   (merged, 0–34 m)
  irte       →  cmems_obs-sdb_glo_phy_irte_my_100m-l4-s2_static   (physics RTE, 0–9 m)
  wk         →  cmems_obs-sdb_glo_phy_wk_my_100m-l4-s2_static     (wave kinematics, 7–34 m)

Output GeoTIFF uses the same sign convention as EMODnet:
    negative values  →  below sea level  (depth in metres as negative number)
    NaN              →  land / no data

Requires: copernicusmarine>=2.4, rasterio, numpy
Credentials: anonymous access works for metadata & open_dataset streaming.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

log = logging.getLogger(__name__)

_DATASET_IDS = {
    "composite": "cmems_obs-sdb_glo_phy_comp_my_100m-l4-s2_static",
    "irte": "cmems_obs-sdb_glo_phy_irte_my_100m-l4-s2_static",
    "it": "cmems_obs-sdb_glo_phy_it_my_100m-l4-s2_static",
    "wk": "cmems_obs-sdb_glo_phy_wk_my_100m-l4-s2_static",
}

MIN_QI_DEFAULT = 3  # only use High / Very High quality pixels


def fetch_cmems_sdb(
    bbox: tuple[float, float, float, float],  # (W, S, E, N) in WGS84
    output_path: str | Path,
    method: str = "composite",
    min_qi: int = MIN_QI_DEFAULT,
) -> Path | None:
    """
    Download CMEMS coastal SDB for *bbox*, mask to QI ≥ *min_qi*, and save
    as a GeoTIFF with negative-depth convention (same as EMODnet).

    Returns the output path on success, None if no valid data in bbox.
    """
    try:
        import copernicusmarine as cm
    except ImportError:
        log.warning("copernicusmarine not installed — skipping CMEMS SDB fetch")
        return None

    dataset_id = _DATASET_IDS.get(method)
    if dataset_id is None:
        raise ValueError(f"Unknown CMEMS SDB method '{method}'. Choose from: {list(_DATASET_IDS)}")

    W, S, E, N = bbox
    log.info("Fetching CMEMS SDB (%s) for bbox (%.3f,%.3f → %.3f,%.3f)", method, W, S, E, N)

    try:
        ds = cm.open_dataset(
            dataset_id=dataset_id,
            minimum_longitude=W,
            maximum_longitude=E,
            minimum_latitude=S,
            maximum_latitude=N,
        )
    except Exception as exc:
        log.warning("CMEMS open_dataset failed: %s", exc)
        return None

    # height variable: negative = below sea level
    height = ds["height"].values.astype(np.float32)

    # Quality mask
    if "Quality_indicator" in ds:
        qi = ds["Quality_indicator"].values
        height = np.where(qi >= min_qi, height, np.nan)

    # Keep only marine pixels (depth < 0)
    depth_neg = np.where(height < 0, height, np.nan)

    n_valid = int(np.isfinite(depth_neg).sum())
    if n_valid == 0:
        log.warning("CMEMS SDB: no valid pixels (QI≥%d, depth<0) in bbox — skipping", min_qi)
        return None

    log.info(
        "CMEMS SDB: %d valid pixels, depth range %.1f–%.1f m",
        n_valid,
        float(np.nanmin(depth_neg)),
        float(np.nanmax(depth_neg)),
    )

    rows, cols = depth_neg.shape
    transform = from_bounds(W, S, E, N, cols, rows)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(depth_neg, 1)

    log.info("CMEMS SDB saved → %s", out_path)
    return out_path


def reproject_cmems_to_s2(
    cmems_path: str | Path,
    s2_reference_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Reproject and resample the 100m CMEMS SDB GeoTIFF to exactly match the
    10m Sentinel-2 reference grid (same CRS, transform, size).
    """
    with rasterio.open(s2_reference_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(cmems_path) as src:
        data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=1,
        dtype="float32",
        crs=dst_crs,
        transform=dst_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)

    log.info("CMEMS SDB reprojected to 10 m S2 grid → %s", out_path)
    return out_path


def blend_depth_priors(
    emodnet_10m: np.ndarray,
    cmems_10m: np.ndarray | None,
    cmems_weight: float = 0.4,
) -> np.ndarray:
    """
    Blend EMODnet (primary, 0.6 weight) with CMEMS SDB (secondary, 0.4 weight)
    where both are valid.  Falls back to whichever source is available alone.

    Both arrays must be on the same grid (negative depth convention).
    Returns blended depth array (negative convention, NaN = no data).
    """
    if cmems_10m is None:
        return emodnet_10m.copy()

    e_valid = np.isfinite(emodnet_10m)
    c_valid = np.isfinite(cmems_10m)
    both = e_valid & c_valid

    blended = np.full_like(emodnet_10m, np.nan)
    w_e = 1.0 - cmems_weight
    blended[both] = w_e * emodnet_10m[both] + cmems_weight * cmems_10m[both]
    blended[e_valid & ~both] = emodnet_10m[e_valid & ~both]  # EMODnet only
    blended[c_valid & ~both] = cmems_10m[c_valid & ~both]  # CMEMS only

    n_both = int(both.sum())
    n_eonly = int((e_valid & ~both).sum())
    n_conly = int((c_valid & ~both).sum())
    log.info(
        "Depth prior blend: %d both (%.0f/%.0f%%), %d EMODnet-only, %d CMEMS-only",
        n_both,
        w_e * 100,
        cmems_weight * 100,
        n_eonly,
        n_conly,
    )
    return blended
