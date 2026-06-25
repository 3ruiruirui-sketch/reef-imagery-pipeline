"""
sh_downloader.py — Sentinel Hub Process API downloader for Sentinel-2 L2A.

Drop-in alternative to the Earth Search / Planetary Computer STAC path.

Supports two Sentinel Hub endpoints (the same Process/Catalog/Statistical API)
with automatic fallback when the commercial SH trial quota runs dry:

  * "commercial" — services.sentinel-hub.com  (current paid/trial SH)
  * "cdse"       — sh.dataspace.copernicus.eu (Copernicus Data Space, free
                   ~30k PU/month; needs a free OAuth client from
                   shapps.dataspace.copernicus.eu/dashboard)

Credentials (read from environment, no hard-coding):

  Commercial SH :  SH_CLIENT_ID, SH_CLIENT_SECRET
  CDSE SH       :  CDSE_CLIENT_ID, CDSE_CLIENT_SECRET
                   (separate from SH_CLIENT_* — different OAuth realm)

Key functions
-------------
    download_scene(bbox, date_range, bands, output_path, endpoint="auto")
        Download a full scene as a float32 GeoTIFF.

    download_patch(lon, lat, size_m, date_range, bands, output_path, endpoint="auto")
        Download a square patch centred on (lon, lat).

    search_scenes(bbox, date_range, max_cloud, collection, endpoint="auto")
        Return a list of available scene metadata (no download).

    get_stats(bbox, date_range, bands, endpoint="auto")
        Monthly band statistics via the Statistical API.

The `endpoint` argument selects which SH backend to use. The default "auto"
tries the commercial endpoint first and falls back to CDSE on any HTTP error
when both credential pairs are present in the environment. With a single
credential pair (commercial OR cdse), the available endpoint is used directly
without fallback.

Usage
-----
    from src.sh_downloader import download_patch

    # Endpoint picked automatically; falls back to CDSE when SH trial runs out
    download_patch(
        lon=-8.2102, lat=37.0691,
        size_m=1280,
        date_range=("2024-09-01", "2024-09-30"),
        bands=["B02", "B03", "B04", "B08"],
        output_path="outputs/reef_patches/pedra_eulalia_2024-09.tif",
    )

    # Force the free CDSE endpoint (skip commercial entirely)
    download_patch(..., endpoint="cdse")
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_bounds

log = logging.getLogger(__name__)

EndpointName = Literal["commercial", "cdse", "auto"]
ConcreteEndpoint = Literal["commercial", "cdse"]


@dataclass
class _TokenEntry:
    """One cached OAuth bearer token + its expiry time."""
    token: str | None = None
    expires_at: float = 0.0


# Sentinel-2 band → BOA reflectance (in FLOAT32 [0, 1] from Process API)
S2_BANDS_L2A = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
    "SCL", "AOT", "WVP", "dataMask",
]

_DEFAULT_BANDS = ["B02", "B03", "B04", "B08"]   # blue, green, red, NIR


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint registry
# ─────────────────────────────────────────────────────────────────────────────

_ENDPOINTS: dict[str, dict[str, str]] = {
    "commercial": {
        "base_url":   "https://services.sentinel-hub.com",
        "token_url":  "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token",
        "id_env":     "SH_CLIENT_ID",
        "secret_env": "SH_CLIENT_SECRET",
    },
    "cdse": {
        "base_url":   "https://sh.dataspace.copernicus.eu",
        "token_url":  "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        "id_env":     "CDSE_CLIENT_ID",
        "secret_env": "CDSE_CLIENT_SECRET",
    },
}


def _endpoint_urls(name: ConcreteEndpoint) -> str:
    """Return the base URL for the named endpoint."""
    return _ENDPOINTS[name]["base_url"]


def _available_endpoints() -> list[str]:
    """Return names of endpoints whose credentials are present in the environment."""
    return [
        name for name, cfg in _ENDPOINTS.items()
        if os.environ.get(cfg["id_env"]) and os.environ.get(cfg["secret_env"])
    ]


# Per-endpoint token cache — both endpoints can coexist in the same process.
_token_cache: dict[str, _TokenEntry] = {
    name: _TokenEntry() for name in _ENDPOINTS
}


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

def _get_token(endpoint: ConcreteEndpoint) -> str:
    """Return a valid OAuth2 Bearer token for the given endpoint, refreshing
    if within 60 s of expiry."""
    cfg = _ENDPOINTS[endpoint]
    cache = _token_cache[endpoint]

    now = time.time()
    if cache.token and now < cache.expires_at - 60:
        return cache.token

    client_id     = os.environ.get(cfg["id_env"])
    client_secret = os.environ.get(cfg["secret_env"])
    if not client_id or not client_secret:
        raise RuntimeError(
            f"Endpoint '{endpoint}' requires {cfg['id_env']} and "
            f"{cfg['secret_env']} in the environment. "
            "Load them with: set -a; source .env; set +a   (or in Python: "
            "from dotenv import load_dotenv; load_dotenv())"
        )

    resp = requests.post(
        cfg["token_url"],
        data={"grant_type": "client_credentials",
              "client_id": client_id,
              "client_secret": client_secret},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    cache.token      = data["access_token"]
    cache.expires_at = now + data.get("expires_in", 3600)
    return cache.token


def _headers(endpoint: ConcreteEndpoint) -> dict:
    return {"Authorization": f"Bearer {_get_token(endpoint)}",
            "Content-Type": "application/json"}


def _post_with_fallback(
    path: str,
    json: dict,
    *,
    endpoint: EndpointName,
    timeout: int,
) -> requests.Response:
    """POST to a SH endpoint, falling back to the next endpoint on any HTTP
    error (or request exception) when `endpoint == 'auto'` and multiple
    endpoints are configured.

    Returns the successful Response. Raises the last response error if every
    endpoint fails.
    """
    endpoints = _fallback_chain(endpoint)

    last_err: Exception | None = None
    for ep in endpoints:
        base = _endpoint_urls(ep)
        url = f"{base}{path}" if path.startswith("/") else f"{base}/{path}"
        try:
            resp = requests.post(url, headers=_headers(ep), json=json, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            log.warning("Endpoint '%s' request error: %s", ep, exc)
            continue

        if resp.ok:
            if ep != endpoints[0]:
                log.info("Fell back from '%s' to '%s' successfully",
                         endpoints[0], ep)
            return resp

        # In auto mode with fallback endpoints available, retry on any HTTP
        # error — the next endpoint may be in a different realm entirely.
        # On an explicit endpoint, only retry on transient errors (auth/quota).
        is_auto_with_fallback = len(endpoints) > 1
        is_retryable_on_explicit = resp.status_code in (401, 403, 429)
        if is_auto_with_fallback or is_retryable_on_explicit:
            log.warning(
                "Endpoint '%s' HTTP %d — %s. Body: %.200s",
                ep, resp.status_code,
                "trying fallback" if is_auto_with_fallback else "non-retryable",
                resp.text,
            )
            last_err = RuntimeError(f"{ep} HTTP {resp.status_code}")
            if is_auto_with_fallback:
                # Invalidate the failed endpoint's cached token so the next
                # call doesn't re-pay the roundtrip with a known-bad token.
                _token_cache[ep].token      = None
                _token_cache[ep].expires_at = 0.0
                continue
            # Explicit endpoint with retryable code — still only one option.
            raise last_err

        # Non-retryable error on explicit endpoint — surface immediately
        raise RuntimeError(
            f"{ep} HTTP {resp.status_code}: {resp.text[:400]}"
        )

    raise RuntimeError(
        f"All Sentinel Hub endpoints failed for {path}: {last_err}"
    )


def _fallback_chain(endpoint: EndpointName) -> list[ConcreteEndpoint]:
    """Build the ordered list of endpoints to try."""
    if endpoint != "auto":
        if endpoint not in _available_endpoints():
            raise RuntimeError(
                f"Endpoint '{endpoint}' requested but its credentials "
                f"({_ENDPOINTS[endpoint]['id_env']}) are not set in the environment."
            )
        return [endpoint]

    available = _available_endpoints()
    if not available:
        raise RuntimeError(
            "No Sentinel Hub credentials found. Set SH_CLIENT_ID/SH_CLIENT_SECRET "
            "(commercial) and/or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET (free CDSE). "
            "Load them with: set -a; source .env; set +a   (or in Python: "
            "from dotenv import load_dotenv; load_dotenv())"
        )
    # Prefer the order they appear in _ENDPOINTS so "commercial" is tried first.
    return [ep for ep in ("commercial", "cdse") if ep in available]


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
    endpoint: EndpointName = "auto",
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

    endpoints = _fallback_chain(endpoint)
    items: list[dict] = []
    last_status = 0
    network_failures: list[str] = []
    for ep in endpoints:
        base = _endpoint_urls(ep)
        try:
            resp = requests.post(
                f"{base}/api/v1/catalog/1.0.0/search",
                headers=_headers(ep), json=payload, timeout=30,
            )
        except requests.RequestException as exc:
            log.warning("Catalog search on '%s' failed: %s", ep, exc)
            network_failures.append(f"{ep}: {exc}")
            continue
        last_status = resp.status_code
        if resp.ok:
            items = resp.json().get("features", [])
            break
        log.warning("Catalog search '%s' HTTP %d", ep, resp.status_code)
        # Invalidate the cached token for this endpoint so the next call
        # doesn't re-use a token that just caused a 401/403.
        _token_cache[ep].token = None
        _token_cache[ep].expires_at = 0.0

    # If we never got an OK response from any endpoint, surface the failure
    # rather than silently returning an empty scene list — the caller cannot
    # otherwise distinguish "no scenes in date range" from "network is down
    # on every configured endpoint" or "catalog access is denied".
    if not items and last_status == 0 and network_failures:
        raise RuntimeError(
            "Catalog search failed on all endpoints (network errors): "
            + "; ".join(network_failures)
        )
    if not items and last_status >= 400:
        log.warning(
            "Catalog search HTTP %d across endpoints — trial may restrict "
            "catalog access; use date_range directly in download functions.",
            last_status,
        )
        return []

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
    endpoint: EndpointName = "auto",
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
    endpoint : 'auto' (default, tries commercial then CDSE), 'commercial',
        or 'cdse'.

    Returns
    -------
    Path to the written GeoTIFF.

    Raises
    ------
    RuntimeError : if the Process API returns an error on all endpoints.
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
        "Process API (%s): %s  bands=%s  grid=%d×%d  → %s",
        endpoint, collection, bands, width, height, output_path,
    )
    resp = _post_with_fallback(
        "/api/v1/process", payload, endpoint=endpoint, timeout=timeout,
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
    endpoint: EndpointName = "auto",
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
        endpoint=endpoint,
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
    endpoint: EndpointName = "auto",
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

    resp = _post_with_fallback(
        "/api/v1/statistics", payload, endpoint=endpoint, timeout=60,
    )

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
