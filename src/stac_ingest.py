"""STAC ingestion helpers for Sentinel-2 data.

This module is intentionally lightweight and reusable by scripts that need
Sentinel-2 scene discovery, asset extraction, and cloud-filtered scene ranking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import planetary_computer as pc
from pystac import Asset, Item
from pystac_client import Client

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_S2_COLLECTION = "sentinel-2-l2a"
STAC_URL_PRIORITY = [EARTH_SEARCH_STAC_URL, PC_STAC_URL]


def open_stac_catalog(url: str = EARTH_SEARCH_STAC_URL) -> Client:
    """Open a STAC catalog and apply signing if required."""
    modifier = pc.sign_inplace if "planetarycomputer.microsoft.com" in url else None
    return Client.open(url, modifier=modifier)


def normalize_datetime_range(
    date_range: Union[str, Tuple[str, str], Tuple[datetime, datetime]]
) -> str:
    if isinstance(date_range, str):
        return date_range
    if len(date_range) != 2:
        raise ValueError("date_range must be a string or a 2-tuple")
    start, end = date_range
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d")
    return f"{start}/{end}"


def search_sentinel2_scenes(
    lat: float,
    lon: float,
    date_range: Union[str, Tuple[str, str], Tuple[datetime, datetime]],
    max_cloud_cover: float = 25.0,
    catalog_url: str | None = None,
    collection: str = DEFAULT_S2_COLLECTION,
    limit: int = 10,
) -> List[Item]:
    """Search Sentinel-2 L2A scenes for a point, date range, and cloud filter.

    If catalog_url is None, this function tries the preferred STAC endpoints in
    order and returns the first successful result.
    """
    urls = [catalog_url] if catalog_url else STAC_URL_PRIORITY
    datetime_range = normalize_datetime_range(date_range)
    last_error: Exception | None = None

    for url in urls:
        try:
            catalog = open_stac_catalog(url)
            search = catalog.search(
                collections=[collection],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            )
            items = list(search.items())
            return items[:limit]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(
        f"STAC scene search failed for all endpoints {urls}: {last_error}"
    )


def choose_least_cloudy(items: Iterable[Item]) -> Optional[Item]:
    """Choose the wind-sense cloudiest scene that has the lowest STAC cloud cover."""
    items = [item for item in items]
    if not items:
        return None
    return min(items, key=lambda item: item.properties.get("eo:cloud_cover", 100.0))


def get_asset_hrefs(item: Item, asset_keys: Iterable[str]) -> Dict[str, str]:
    """Return a mapping of asset key -> signed asset URL for a STAC item."""
    hrefs: Dict[str, str] = {}

    # Build a case-insensitive map of available asset keys
    available = {k.lower(): v for k, v in item.assets.items()}

    # Common alias mapping for Sentinel-2 band names in different STAC catalogs
    alias_map = {
        "b02": ["blue", "b02"],
        "b03": ["green", "b03"],
        "b04": ["red", "b04"],
        "b08": ["nir", "b08", "b8a"],
        "b11": ["swir16", "b11"],
        "b12": ["swir22", "b12"],
    }

    for key in asset_keys:
        k_low = key.lower()

        # Direct match
        if k_low in available:
            hrefs[key] = available[k_low].href
            continue

        # Try aliases (e.g., 'B02' -> 'blue')
        aliases = alias_map.get(k_low, [k_low])
        found = False
        for a in aliases:
            if a in available:
                hrefs[key] = available[a].href
                found = True
                break

        if found:
            continue

        # Fallback: look for any asset key that contains the band name
        for a_key, a_asset in available.items():
            if k_low in a_key:
                hrefs[key] = a_asset.href
                break

    return hrefs


def scene_summary(item: Item) -> Dict[str, Any]:
    """Return a lightweight metadata summary for a Sentinel-2 STAC item."""
    return {
        "id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "sun_elevation": item.properties.get("view:sun_elevation"),
        "platform": item.properties.get("platform"),
        "collection": item.collection_id,
        "assets": sorted(item.assets.keys()),
    }
