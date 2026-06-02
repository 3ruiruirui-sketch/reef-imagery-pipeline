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


def open_stac_catalog(url: str = PC_STAC_URL) -> Client:
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
    catalog_url: str = PC_STAC_URL,
    collection: str = DEFAULT_S2_COLLECTION,
    limit: int = 10,
) -> List[Item]:
    """Search Sentinel-2 L2A scenes for a point, date range, and cloud filter."""
    catalog = open_stac_catalog(catalog_url)
    datetime_range = normalize_datetime_range(date_range)

    search = catalog.search(
        collections=[collection],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    items = list(search.get_items())
    return items[:limit]


def choose_least_cloudy(items: Iterable[Item]) -> Optional[Item]:
    """Choose the wind-sense cloudiest scene that has the lowest STAC cloud cover."""
    items = [item for item in items]
    if not items:
        return None
    return min(items, key=lambda item: item.properties.get("eo:cloud_cover", 100.0))


def get_asset_hrefs(item: Item, asset_keys: Iterable[str]) -> Dict[str, str]:
    """Return a mapping of asset key -> signed asset URL for a STAC item."""
    hrefs: Dict[str, str] = {}
    for key in asset_keys:
        asset = item.assets.get(key)
        if asset is None:
            continue
        hrefs[key] = asset.href
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
