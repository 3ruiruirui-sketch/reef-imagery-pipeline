"""
sentinel1_roughness.py — Sentinel-1 SAR surface roughness for Algarve reef sites.

Provides three functions used by orchestrator_run.py to apply an optional
sea-state penalty to visibility scores:

  roughness_from_sigma0()   — dual-pol roughness proxy (Ulaby et al. 1986)
  extract_sigma0_at_point() — extract VV/VH sigma0 from a STAC COG item
  search_stac_s1_scenes()   — search Element84 Earth Search for S1 GRD scenes

All functions are best-effort: any exception is caught by the caller's
try/except block and the penalty is silently skipped.

Data sources:
  Primary:  Element84 Earth Search STAC (public S1 GRD COG)
            https://earth-search.aws.element84.com/v1
  Fallback: Copernicus CDSE OData API
            https://catalogue.dataspace.copernicus.eu/odata/v1/Products
"""

import math

# Element84 Earth Search STAC endpoint and S1 collection name.
_E84_STAC_URL = "https://earth-search.aws.element84.com/v1"
_E84_S1_COLLECTION = "sentinel-1-grd"

# Roughness thresholds (log10 VV/VH ratio).
ROUGHNESS_CALM_THRESHOLD = 0.05  # below → calm, good diving conditions
ROUGHNESS_ROUGH_THRESHOLD = 0.25  # above → rough sea, poor visibility


def roughness_from_sigma0(sigma0_vv: float, sigma0_vh: float) -> dict:
    """
    Compute surface roughness proxy from dual-polarisation sigma0 values.

    Cross-ratio roughness (Ulaby et al. 1986):
        roughness = log10(VV / VH)

    Physical interpretation:
        Low VV/VH → calm specular surface → good diving visibility window.
        High VV/VH → wind-roughened surface → poor visibility.

    Returns dict with keys: roughness, sea_state, vv_linear, vh_linear,
    vv_db, vh_db.  On invalid input (non-positive sigma0) roughness is None
    and sea_state is "unknown".
    """
    if sigma0_vv <= 0 or sigma0_vh <= 0:
        return {
            "roughness": None,
            "sea_state": "unknown",
            "vv": sigma0_vv,
            "vh": sigma0_vh,
        }

    roughness = math.log10(sigma0_vv / sigma0_vh)

    if roughness < ROUGHNESS_CALM_THRESHOLD:
        sea_state = "calm"
    elif roughness < ROUGHNESS_ROUGH_THRESHOLD:
        sea_state = "moderate"
    else:
        sea_state = "rough"

    return {
        "roughness": round(roughness, 4),
        "sea_state": sea_state,
        "vv_linear": round(sigma0_vv, 6),
        "vh_linear": round(sigma0_vh, 6),
        "vv_db": round(10 * math.log10(sigma0_vv), 2),
        "vh_db": round(10 * math.log10(sigma0_vh), 2),
    }


def search_stac_s1_scenes(
    lon: float,
    lat: float,
    year: int,
    month_start: int,
    month_end: int,
    max_results: int = 5,
) -> list:
    """
    Search Element84 Earth Search STAC for Sentinel-1 IW GRD scenes.

    Returns a list of simplified item dicts (id, date, assets).
    Returns empty list if pystac_client is not installed or the search fails.
    """
    try:
        from pystac_client import Client
    except ImportError:
        return []

    try:
        catalog = Client.open(_E84_STAC_URL)
        start_dt = f"{year}-{month_start:02d}-01T00:00:00Z"
        end_dt = f"{year}-{month_end:02d}-28T23:59:59Z"
        search = catalog.search(
            collections=[_E84_S1_COLLECTION],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=f"{start_dt}/{end_dt}",
            max_items=max_results,
        )
        return [
            {
                "id": it.id,
                "date": it.datetime.strftime("%Y-%m-%d") if it.datetime else "unknown",
                "assets": {k: v.href for k, v in it.assets.items()},
                "properties": dict(it.properties),
            }
            for it in search.items()
        ]
    except Exception:
        return []


def extract_sigma0_at_point(item: dict, lon: float, lat: float):
    """
    Extract VV/VH sigma0 (linear scale) at a geographic point from a STAC item.

    Uses rasterio to read a 3×3 pixel patch from the GRD COG at (lon, lat) and
    converts amplitude to power: sigma0 = amplitude² / 1e8.

    Returns dict with keys 'vv' and 'vh' (linear scale), or None on failure.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject

        assets = item.get("assets", {})
        vv_href = assets.get("vv") or assets.get("VV") or assets.get("measurement_vv")
        vh_href = assets.get("vh") or assets.get("VH") or assets.get("measurement_vh")
        if not vv_href or not vh_href:
            return None

        res = 0.0001  # ~10 m in degrees
        dst_crs = "EPSG:4326"
        dst_transform = from_origin(lon - 1.5 * res, lat + 1.5 * res, res, res)

        env = rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        )
        results = {}
        for pol, href in [("vv", vv_href), ("vh", vh_href)]:
            with env, rasterio.open(href) as src:
                dst_data = np.zeros((3, 3), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_data,
                    src_transform=None,
                    src_crs=None,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                data_pos = dst_data[dst_data > 0]
                if len(data_pos) == 0:
                    results[pol] = None
                else:
                    amplitude_mean = float(np.mean(data_pos))
                    results[pol] = (amplitude_mean**2) / 1e8

        if results.get("vv") and results.get("vh"):
            return results
        return None

    except Exception:
        return None
