"""
sh_downloader.py — Sentinel Hub Process API downloader for Sentinel-2 L2A.

Drop-in alternative to the Earth Search / Planetary Computer STAC path.
Reads credentials from environment variables:
    SH_CLIENT_ID, SH_CLIENT_SECRET

Key functions
-------------
    download_scene(bbox, date_range, bands, output_path)
        Download a full scene as a float32 GeoTIFF.

    download_patch(lon, lat, size_m, date_range, bands, output_path)
        Download a square patch centred on (lon, lat).

    search_scenes(bbox, date_range, max_cloud, collection)
        Return a list of available scene metadata (no download).

    get_stats(bbox, date_range, bands)
        Monthly band statistics via the Statistical API.

Usage
-----
    from src.sh_downloader import download_patch, search_scenes

    # Download the clearest S2-L2A 4-band patch over the GPS reef
    download_patch(
        lon=-8.2102, lat=37.0691,
        size_m=1280,
        date_range=("2024-09-01", "2024-09-30"),
        bands=["B02", "B03", "B04", "B08"],
        output_path="outputs/reef_patches/pedra_eulalia_2024-09.tif",
    )
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

log = logging.getLogger(__name__)

_BASE_URL  = "https://services.sentinel-hub.com"
_TOKEN_URL = f"{_BASE_URL}/auth/realms/main/protocol/openid-connect/token"
_PROCESS_URL   = f"{_BASE_URL}/api/v1/process"
_STAT_URL      = f"{_BASE_URL}/api/v1/statistics"
_CATALOG_URL   = f"{_BASE_URL}/api/v1/catalog/1.0.0"

# Sentinel-2 band → BOA reflectance (in FLOAT32 [0, 1] from Process API)
S2_BANDS_L2A = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
    "SCL", "AOT", "WVP", "dataMask",
]

_DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]   # blue, green, red, NIR

# Token cache — avoids a round-trip every call (tokens last 3600 s)
_token_cache: dict = {"token": None, "expires_at": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

def _get_token() -> str:
    """Return a valid OAuth2 Bearer token, refreshing if within 60 s of expiry."""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    client_id     = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "SH_CLIENT_ID and SH_CLIENT_SECRET must be set in the environment. "
            "Load them with: set -a; source .env; set +a"
        )

    resp = requests.post(
        _TOKEN_URL,
        data={"grant_type": "client_credentials",
              "client_id": client_id,
              "client_secret": client_secret},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"]      = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600)
    return _token_cache["token"]


def _headers() -> dict:
    return {"Authorization": f"Bearer {_get_token()}",
            "Content-Type": "application/json"}


# ─────────────────────────────────────────────────────────────────────────────
# Evalscript builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_evalscript(bands: list[str], sample_type: str = "FLOAT32") -> str:
    """Generate a minimal evalscript that returns the requested bands as-is."""
    band_list = '", "'.join(bands)
    n = len(bands)
    returns = ", ".join(f"s.{b}" for b in bands)
    return f'''//VERSION=3
function setup() {{
  return {{
    input: [{{bands: ["{band_list}", "dataMask"]}}],
    output: {{bands: {n}, sampleType: "{sample_type}"}}
  }};
}}
function evaluatePixel(s) {{
  return [{returns}];
}}'''


def _build_evalscript_with_mask(bands: list[str]) -> str:
    """Evalscript that returns band values + a dataMask output (for Stats API)."""
    band_list = '", "'.join(bands)
    n = len(bands)
    returns = ", ".join(f"s.{b}" for b in bands)
    return f'''//VERSION=3
function setup() {{
  return {{
    input: [{{bands: ["{band_list}", "dataMask"], units: "REFLECTANCE"}}],
    output: [
      {{id: "default", bands: {n}}},
      {{id: "dataMask", bands: 1}}
    ]
  }};
}}
function evaluatePixel(s) {{
  return {{
    default: [{returns}],
    dataMask: [s.dataMask]
  }};
}}'''


# ─────────────────────────────────────────────────────────────────────────────
# Scene search (Catalog API)
# ─────────────────────────────────────────────────────────────────────────────

def search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str] | str,
    *,
    max_cloud: float = 20.0,
    collection: str = "sentinel-2-l2a",
    limit: int = 20,
) -> list[dict]:
    """
    Search the Catalog for available scenes.

    Returns list of dicts with keys: id, datetime, cloud_cover, assets.
    Note: trial accounts may return empty list even when Process API works —
    use date_range directly in download functions in that case.
    """
    if isinstance(date_range, tuple):
        dt = f"{date_range[0]}T00:00:00Z/{date_range[1]}T23:59:59Z"
    else:
        dt = date_range

    payload = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": dt,
        "limit": limit,
    }
    resp = requests.post(f"{_CATALOG_URL}/search", headers=_headers(), json=payload, timeout=30)

    if not resp.ok:
        log.warning("Catalog search HTTP %d — trial may restrict catalog access", resp.status_code)
        return []

    items = resp.json().get("features", [])
    results = []
    for item in items:
        props = item["properties"]
        cloud = props.get("eo:cloud_cover", 100.0)
        if cloud <= max_cloud:
            results.append({
                "id":          item["id"],
                "datetime":    props.get("datetime", ""),
                "cloud_cover": cloud,
                "assets":      list(item.get("assets", {}).keys()),
            })
    results.sort(key=lambda x: x["cloud_cover"])
    log.info("Catalog: %d scenes ≤ %.0f%% cloud in %s", len(results), max_cloud, collection)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Process API download
# ─────────────────────────────────────────────────────────────────────────────

def download_scene(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str] | str,
    *,
    bands: list[str] = _DEFAULT_BANDS,
    output_path: str | Path,
    resolution_m: float = 10.0,
    max_cloud: float = 20.0,
    collection: str = "sentinel-2-l2a",
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
) -> Path:
    """
    Download a Sentinel-2 scene as a float32 GeoTIFF via the Process API.

    The mosaic uses the least-cloudy acquisition within the date range — the
    same behaviour as the existing STAC path but served directly from SH.

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max) in WGS-84.
    date_range : ("YYYY-MM-DD", "YYYY-MM-DD") or "YYYY-MM-DD/YYYY-MM-DD".
    bands : list of S2-L2A band names (default: B02 B03 B04 B08).
    output_path : destination GeoTIFF path.
    resolution_m : pixel spacing in metres (default 10 m).
    max_cloud : maximum cloud cover % filter (default 20 %).
    collection : SH collection ID (default 'sentinel-2-l2a').
    mosaicking_order : 'leastCC' (default), 'mostRecent', 'leastRecent'.

    Returns
    -------
    Path to the written GeoTIFF.

    Raises
    ------
    RuntimeError : if the Process API returns an error.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(date_range, tuple):
        time_from = f"{date_range[0]}T00:00:00Z"
        time_to   = f"{date_range[1]}T23:59:59Z"
    else:
        parts = date_range.split("/")
        time_from = parts[0] if "T" in parts[0] else f"{parts[0]}T00:00:00Z"
        time_to   = parts[1] if "T" in parts[1] else f"{parts[1]}T23:59:59Z"

    lon_min, lat_min, lon_max, lat_max = bbox
    deg_per_m   = resolution_m / 111_320
    width  = max(1, round((lon_max - lon_min) / deg_per_m))
    height = max(1, round((lat_max - lat_min) / deg_per_m))

    payload = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": collection,
                "dataFilter": {
                    "timeRange": {"from": time_from, "to": time_to},
                    "maxCloudCoverage": max_cloud,
                    "mosaickingOrder": mosaicking_order,
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default",
                           "format": {"type": "image/tiff",
                                      "parameters": {"sampleType": "FLOAT32"}}}],
        },
        "evalscript": _build_evalscript(bands, "FLOAT32"),
    }

    log.info(
        "Process API: %s  bands=%s  grid=%d×%d  → %s",
        collection, bands, width, height, output_path,
    )
    resp = requests.post(_PROCESS_URL, headers=_headers(), json=payload, timeout=timeout)

    if not resp.ok:
        raise RuntimeError(
            f"Process API HTTP {resp.status_code}: {resp.text[:400]}"
        )

    # Parse the returned TIFF and re-write with correct georeferencing
    with rasterio.open(io.BytesIO(resp.content)) as src:
        data    = src.read()
        profile = src.profile.copy()

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    profile.update(
        driver="GTiff", dtype="float32", count=len(bands),
        crs=CRS.from_epsg(4326), transform=transform,
        width=width, height=height,
        compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        dst.update_tags(
            bands=",".join(bands),
            source=collection,
            date_from=time_from,
            date_to=time_to,
        )

    log.info(
        "Saved: %s  (%d bands, %d×%d px, %.2f MB)",
        output_path, len(bands), width, height, output_path.stat().st_size / 1e6,
    )
    return output_path


def download_patch(
    lon: float,
    lat: float,
    *,
    size_m: float = 1280.0,
    date_range: tuple[str, str] | str,
    bands: list[str] = _DEFAULT_BANDS,
    output_path: str | Path,
    resolution_m: float = 10.0,
    max_cloud: float = 20.0,
) -> Path:
    """
    Download a square patch centred on (lon, lat) with side length size_m metres.

    Convenience wrapper around download_scene() for point-centred patches.
    """
    half_deg = (size_m / 2) / 111_320
    bbox = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
    return download_scene(
        bbox, date_range,
        bands=bands, output_path=output_path,
        resolution_m=resolution_m, max_cloud=max_cloud,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistical API
# ─────────────────────────────────────────────────────────────────────────────

def get_stats(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    *,
    bands: list[str] = _DEFAULT_BANDS,
    interval: str = "P1M",
    max_cloud: float = 20.0,
    resolution_deg: float = 0.0001,
) -> list[dict]:
    """
    Compute monthly (or custom-interval) band statistics via the Statistical API.

    Returns list of dicts:
        { "from": "...", "to": "...", "bands": {"B02": {"mean": 0.04, ...}, ...} }
    """
    payload = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"mosaickingOrder": "leastCC",
                                     "maxCloudCoverage": max_cloud}}],
        },
        "aggregation": {
            "timeRange": {
                "from": f"{date_range[0]}T00:00:00Z",
                "to":   f"{date_range[1]}T23:59:59Z",
            },
            "aggregationInterval": {"of": interval},
            "evalscript": _build_evalscript_with_mask(bands),
            "resx": resolution_deg,
            "resy": resolution_deg,
        },
        "calculations": {
            "default": {"statistics": {"default": {"percentiles": {"k": [10, 25, 50, 75, 90]}}}}
        },
    }

    resp = requests.post(_STAT_URL, headers=_headers(), json=payload, timeout=60)
    resp.raise_for_status()

    results = []
    for interval_data in resp.json().get("data", []):
        itvl = interval_data["interval"]
        band_stats = {}
        raw_bands  = interval_data.get("outputs", {}).get("default", {}).get("bands", {})
        for i, band_name in enumerate(bands):
            key = f"B{i}"
            st  = raw_bands.get(key, {}).get("stats", {})
            band_stats[band_name] = {
                "mean":    round(st.get("mean",    0.0), 6),
                "stdev":   round(st.get("stDev",   0.0), 6),
                "min":     round(st.get("min",     0.0), 6),
                "max":     round(st.get("max",     0.0), 6),
                "p10":     round(st.get("percentiles", {}).get("10.0", 0.0), 6),
                "p50":     round(st.get("percentiles", {}).get("50.0", 0.0), 6),
                "p90":     round(st.get("percentiles", {}).get("90.0", 0.0), 6),
                "n_valid": st.get("sampleCount", 0) - st.get("noDataCount", 0),
            }
        results.append({
            "from":  itvl["from"][:10],
            "to":    itvl["to"][:10],
            "bands": band_stats,
        })

    log.info("Stats: %d intervals for %s", len(results), bands)
    return results
