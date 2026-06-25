"""
s2_patch_extractor.py — Sentinel-2 4-band patch extraction for reef classification.

For each labelled point (lat/lon + class), this module:
  1. Finds the best (lowest cloud) Sentinel-2 L2A scene via Earth Search STAC.
  2. Downloads the 4 relevant bands: B02 (blue), B03 (green), B04 (red), B08 (NIR).
  3. Reads a 128×128-pixel window centred on the point (1.28 km × 1.28 km footprint).
  4. Applies per-channel min-max normalisation using Algarve-calibrated clip percentiles.
  5. Saves the result as a (4, 128, 128) float32 numpy array (.npy).

Patch size rationale: 128 Sentinel-2 pixels ≈ 1.28 km — matches the EMODnet grid
resolution at which reef/sand labels are derived (~115 m). A smaller patch would
oscillate between reef and background sand within a single EMODnet cell.

Band order in the 4-channel array: [B02, B03, B04, B08]

Usage
-----
    from src.s2_patch_extractor import build_dataset, extract_patch

    build_dataset(
        labels_path="outputs/reef_habitat/labels/label_manifest.json",
        output_dir="outputs/reef_habitat/patches",
        scene_id="S2B_29SNB_20240930_0_L2A",   # or None for auto-select
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.windows import Window

log = logging.getLogger(__name__)

# ── Spectral configuration ─────────────────────────────────────────────────────
BANDS = ["B02", "B03", "B04", "B08"]  # blue, green, red, NIR (10 m resolution)
PATCH_PX = 128  # output patch size in pixels
S2_RES_M = 10.0  # Sentinel-2 native resolution (m)

# Earth Search (AWS) uses descriptive names; Planetary Computer uses band codes.
# This table maps our canonical band IDs to the asset key used by each catalog.
_BAND_ALIASES: dict[str, list[str]] = {
    "B02": ["blue", "B02", "B2"],
    "B03": ["green", "B03", "B3"],
    "B04": ["red", "B04", "B4"],
    "B08": ["nir", "nir08", "B08", "B8", "B8A"],
}

# Per-channel normalisation clip percentiles (derived from clear-water Algarve scenes)
# Values are BOA reflectance × 10000.  Anything below p_low is clipped to 0, above
# p_high is clipped to 1.  These preserve contrast in the 0–25 m depth range.
_CLIP_LOW = {"B02": 0, "B03": 0, "B04": 0, "B08": 0}
_CLIP_HIGH = {"B02": 2500, "B03": 2000, "B04": 1800, "B08": 3000}

# ── STAC endpoint ──────────────────────────────────────────────────────────────
_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
_S2_COLLECTION = "sentinel-2-l2a"

# Default date window: capture full clear-water season around the validated scene
_DEFAULT_DATE_RANGE = "2024-06-01/2024-12-31"


# ─────────────────────────────────────────────────────────────────────────────
# 1. STAC scene discovery
# ─────────────────────────────────────────────────────────────────────────────


def _find_best_scene(lat: float, lon: float, date_range: str, max_cloud: float = 5.0) -> dict | None:
    """
    Return metadata dict for the clearest Sentinel-2 scene covering (lon, lat).
    Uses pystac-client if available; falls back to raw STAC API otherwise.
    Returns None if nothing found.
    """
    try:
        from pystac_client import Client

        cat = Client.open(_EARTH_SEARCH)
        results = cat.search(
            collections=[_S2_COLLECTION],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
            max_items=5,
        )
        items = list(results.items())
        if not items:
            return None
        best = items[0]
        return {
            "id": best.id,
            "cloud_cover": best.properties.get("eo:cloud_cover", 99),
            "datetime": best.properties.get("datetime", ""),
            "assets": {k: v.href for k, v in best.assets.items()},
        }
    except Exception as exc:
        log.warning("pystac-client search failed (%s) — falling back to raw API", exc)

    # Raw fallback
    try:
        url = f"{_EARTH_SEARCH}/search"
        payload = {
            "collections": [_S2_COLLECTION],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": date_range,
            "query": {"eo:cloud_cover": {"lt": max_cloud}},
            "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
            "limit": 5,
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return None
        best = features[0]
        return {
            "id": best["id"],
            "cloud_cover": best["properties"].get("eo:cloud_cover", 99),
            "datetime": best["properties"].get("datetime", ""),
            "assets": {k: v["href"] for k, v in best["assets"].items()},
        }
    except Exception as exc2:
        log.error("Raw STAC API search also failed: %s", exc2)
        return None


def _find_scene_by_id(scene_id: str) -> dict | None:
    """Fetch a specific STAC item by scene_id."""
    try:
        from pystac_client import Client

        cat = Client.open(_EARTH_SEARCH)
        item = cat.get_collection(_S2_COLLECTION).get_item(scene_id)
        if item is None:
            raise ValueError("Not found")
        return {
            "id": item.id,
            "cloud_cover": item.properties.get("eo:cloud_cover", 99),
            "datetime": item.properties.get("datetime", ""),
            "assets": {k: v.href for k, v in item.assets.items()},
        }
    except Exception:
        pass

    try:
        url = f"{_EARTH_SEARCH}/collections/{_S2_COLLECTION}/items/{scene_id}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        best = resp.json()
        return {
            "id": best["id"],
            "cloud_cover": best["properties"].get("eo:cloud_cover", 99),
            "datetime": best["properties"].get("datetime", ""),
            "assets": {k: v["href"] for k, v in best["assets"].items()},
        }
    except Exception as exc:
        log.error("Scene lookup by ID failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Band download and patch extraction
# ─────────────────────────────────────────────────────────────────────────────


def _band_url(assets: dict, band: str) -> str | None:
    """Resolve asset URL for *band* (e.g. 'B02') across Earth Search / PC naming schemes."""
    aliases = _BAND_ALIASES.get(band, [band])
    for alias in aliases:
        if alias in assets:
            return assets[alias]
        # Case-insensitive fallback
        for key, href in assets.items():
            if key.lower() == alias.lower():
                return href
    return None


def _download_band_window(
    href: str,
    lon: float,
    lat: float,
    patch_px: int,
) -> np.ndarray | None:
    """
    Open a COG band from *href*, extract a *patch_px* × *patch_px* window
    centred on (*lon*, *lat*), and return the raw DN array (int16 / uint16).

    Returns None if the point falls outside the raster or the download fails.
    """
    try:
        with rasterio.open(href) as src:
            # Convert geographic coords to pixel coords in the raster's CRS
            if src.crs and src.crs.to_epsg() != 4326:
                from pyproj import Transformer

                xformer = Transformer.from_crs("EPSG:4326", src.crs.to_epsg(), always_xy=True)
                px, py = xformer.transform(lon, lat)
            else:
                px, py = lon, lat

            row_f, col_f = src.index(px, py)

            # Ensure point is inside the raster
            if not (0 <= row_f < src.height and 0 <= col_f < src.width):
                log.debug("Point (%f, %f) outside raster bounds for band %s", lon, lat, href)
                return None

            half = patch_px // 2
            col_off = max(0, col_f - half)
            row_off = max(0, row_f - half)
            # Clamp to raster dimensions
            col_off = min(col_off, src.width - patch_px)
            row_off = min(row_off, src.height - patch_px)

            window = Window.from_slices(
                rows=(row_off, row_off + patch_px),
                cols=(col_off, col_off + patch_px),
            )
            data = src.read(1, window=window, resampling=Resampling.bilinear)

            if data.shape != (patch_px, patch_px):
                # Pad with zeros if at edge
                padded = np.zeros((patch_px, patch_px), dtype=data.dtype)
                padded[: data.shape[0], : data.shape[1]] = data
                data = padded

            return data.astype(np.float32)

    except Exception as exc:
        log.warning("Band download failed for %s: %s", href, exc)
        return None


def _normalise_band(data: np.ndarray, band: str) -> np.ndarray:
    """Clip and rescale band DN to [0, 1] using pre-calibrated clip values."""
    lo = _CLIP_LOW[band]
    hi = _CLIP_HIGH[band]
    clipped = np.clip(data, lo, hi)
    return (clipped - lo) / (hi - lo)


def extract_patch(
    lon: float,
    lat: float,
    assets: dict,
    patch_px: int = PATCH_PX,
) -> np.ndarray | None:
    """
    Extract a normalised (4, patch_px, patch_px) float32 array from *assets*
    for the point (*lon*, *lat*).

    Band order: [B02, B03, B04, B08].

    Returns None if any band cannot be downloaded.
    """
    channels = []
    for band in BANDS:
        href = _band_url(assets, band)
        if href is None:
            log.error("No asset for band %s in scene", band)
            return None
        arr = _download_band_window(href, lon, lat, patch_px)
        if arr is None:
            return None
        channels.append(_normalise_band(arr, band))

    return np.stack(channels, axis=0).astype(np.float32)  # (4, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    scene_id: str | None = None,
    date_range: str = _DEFAULT_DATE_RANGE,
    patch_px: int = PATCH_PX,
    val_fraction: float = 0.2,
    max_cloud: float = 5.0,
) -> dict:
    """
    Extract Sentinel-2 patches for every labelled point in *labels_path*.

    Parameters
    ----------
    labels_path   : Path to label_manifest.json produced by build_reef_dataset.py.
    output_dir    : Root output directory; patches saved to
                    {output_dir}/train/{class}/ and {output_dir}/val/{class}/.
    scene_id      : If given, use this specific scene.  Otherwise auto-select
                    the clearest scene covering each point.
    date_range    : STAC datetime filter string ("YYYY-MM-DD/YYYY-MM-DD").
    patch_px      : Patch side length in pixels (default 128).
    val_fraction  : Fraction of patches reserved for validation.
    max_cloud     : Maximum cloud cover % for auto scene selection.

    Returns
    -------
    dict with counts: n_reef_train, n_reef_val, n_sand_train, n_sand_val, n_failed.
    """
    labels_path = Path(labels_path)
    output_dir = Path(output_dir)

    with open(labels_path) as f:
        manifest = json.load(f)

    labels: list[dict] = manifest["labels"]  # [{lon, lat, class, source}, ...]
    log.info("Loaded %d labels from %s", len(labels), labels_path)

    # Resolve scene (shared across all labels — Algarve fits in one S2 tile)
    if scene_id:
        scene = _find_scene_by_id(scene_id)
        if scene is None:
            raise RuntimeError(f"Scene not found: {scene_id}")
    else:
        # Use the first label point for scene search; all Algarve points share
        # the same Sentinel-2 tile (29SNB)
        ref = labels[0]
        scene = _find_best_scene(ref["lat"], ref["lon"], date_range, max_cloud)
        if scene is None:
            raise RuntimeError(
                f"No clean S2 scene found in {date_range} (cloud<{max_cloud}%) " f"for point ({ref['lat']}, {ref['lon']})"
            )
    log.info(
        "Using scene: %s  cloud=%.2f%%  date=%s",
        scene["id"],
        scene["cloud_cover"],
        scene["datetime"][:10],
    )

    assets = scene["assets"]

    # Setup output directories
    for split in ("train", "val"):
        for cls in ("reef", "sand"):
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counters = {"n_reef_train": 0, "n_reef_val": 0, "n_sand_train": 0, "n_sand_val": 0, "n_failed": 0}
    class_counts: dict[str, int] = {"reef": 0, "sand": 0}

    # Separate by class, shuffle reproducibly, then assign train/val
    rng = np.random.default_rng(seed=42)
    reef_labels = [l for l in labels if l["class"] == "reef"]
    sand_labels = [l for l in labels if l["class"] == "sand"]

    def _extract_for_split(pts: list[dict], cls: str) -> None:
        rng.shuffle(pts)
        n_val = max(1, round(len(pts) * val_fraction))
        for i, pt in enumerate(pts):
            split = "val" if i < n_val else "train"
            patch = extract_patch(pt["lon"], pt["lat"], assets, patch_px)
            if patch is None:
                log.warning("Failed to extract patch for %s (%f, %f)", cls, pt["lat"], pt["lon"])
                counters["n_failed"] += 1
                continue
            idx = class_counts[cls]
            fname = f"{cls}_{idx:05d}.npy"
            np.save(output_dir / split / cls / fname, patch)
            class_counts[cls] += 1
            key = f"n_{cls}_{split}"
            counters[key] = counters.get(key, 0) + 1
            log.debug("Saved %s/%s/%s  shape=%s", split, cls, fname, patch.shape)

    _extract_for_split(reef_labels, "reef")
    _extract_for_split(sand_labels, "sand")

    summary = {
        **counters,
        "scene_id": scene["id"],
        "scene_date": scene["datetime"][:10],
        "patch_px": patch_px,
        "n_channels": 4,
        "bands": BANDS,
    }
    summary_path = output_dir / "patch_manifest.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "Patch extraction complete: reef=%d+%d  sand=%d+%d  failed=%d  → %s",
        counters["n_reef_train"],
        counters["n_reef_val"],
        counters["n_sand_train"],
        counters["n_sand_val"],
        counters["n_failed"],
        output_dir,
    )
    return summary
