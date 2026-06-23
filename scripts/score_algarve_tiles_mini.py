#!/usr/bin/env python3
"""
score_algarve_tiles_mini.py — Coast-wide reef-probability inference.

Slides a trained patch classifier (ResNet-18, 4-channel Sentinel-2 input) over a
sliding-window grid covering the Algarve bbox and produces:

  1. ``algarve_reef_prob.tif``     — single-band float32 GeoTIFF, georeferenced,
                                     values in [0, 1], one probability per tile
                                     footprint (broadcast across tile pixels).
  2. ``algarve_reef_hotspots.geojson`` — top-K highest-probability tiles as
                                     lon/lat point features.
  3. ``scoring_log.txt``           — human-readable run summary.

This is a NEW capability: the baseline pipeline only classifies the N=15 labelled
patches.  Living in its own ``*_mini.py`` script keeps ``src/s2_patch_extractor.py``
and ``src/reef_classifier.py`` strictly read-only.

Usage
-----
    python scripts/score_algarve_tiles_mini.py \\
        --checkpoint outputs/reef_habitat_mini/model/reef_cnn_best.pth \\
        --scene-id S2A_29SNB_20230901_0_L2A \\
        --bbox -8.6 36.85 -7.7 37.30 \\
        --tile 128 --stride 64 \\
        --threshold 0.5 \\
        --top-hotspots 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds
from rasterio.windows import Window

# Add project root so ``from src.reef_classifier import ...`` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported (NOT copied) from baseline.
from src.reef_classifier import load_model, predict_single

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("score_mini")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel-2 native resolution used for the 10 m bands (B02/B03/B04/B08).
S2_RES_M = 10.0

# Earth Search STAC endpoint (same as src/s2_patch_extractor.py).
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Band order — MUST match the order baked into the checkpoint (B02/B03/B04/B08).
BANDS = ["B02", "B03", "B04", "B08"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers duplicated inline from src/s2_patch_extractor.py (ATTRIBUTION)
# ─────────────────────────────────────────────────────────────────────────────
#
# The two functions below are deliberately inlined here with attribution. They
# are tightly coupled to the s2_patch_extractor module's internal state and
# output paths, so importing them directly is not safe. Replicating ~20 lines
# preserves the invariant "this script never edits src/".
#
# SOURCE: src/s2_patch_extractor.py — _band_url() and _normalise_band()
#         (see lines 50–62 and 170–180, 238–243 of that file).
# ─────────────────────────────────────────────────────────────────────────────

# Earth Search uses descriptive names; Planetary Computer uses band codes.
# Verbatim copy of _BAND_ALIASES from src/s2_patch_extractor.py.
_BAND_ALIASES: dict[str, list[str]] = {
    "B02": ["blue",  "B02", "B2"],
    "B03": ["green", "B03", "B3"],
    "B04": ["red",   "B04", "B4"],
    "B08": ["nir",   "nir08", "B08", "B8", "B8A"],
}

# Per-channel clip percentiles (BOA reflectance × 10000).  Verbatim copy of
# _CLIP_LOW / _CLIP_HIGH from src/s2_patch_extractor.py.
_CLIP_LOW  = {"B02": 0,    "B03": 0,    "B04": 0,    "B08": 0}
_CLIP_HIGH = {"B02": 2500, "B03": 2000, "B04": 1800, "B08": 3000}


def _band_url(assets: dict, band: str) -> Optional[str]:
    """
    Resolve asset URL for *band* (e.g. 'B02') across Earth Search / PC naming.

    ATTRIBUTION: copied verbatim from src/s2_patch_extractor._band_url() to keep
    src/ unchanged. See src/s2_patch_extractor.py:170.
    """
    aliases = _BAND_ALIASES.get(band, [band])
    for alias in aliases:
        if alias in assets:
            return assets[alias]
        for key, href in assets.items():
            if key.lower() == alias.lower():
                return href
    return None


def _normalise_band(data: np.ndarray, band: str) -> np.ndarray:
    """
    Clip and rescale band DN to [0, 1] using pre-calibrated clip values.

    ATTRIBUTION: copied verbatim from src/s2_patch_extractor._normalise_band()
    to keep src/ unchanged. See src/s2_patch_extractor.py:238.
    """
    lo = _CLIP_LOW[band]
    hi = _CLIP_HIGH[band]
    clipped = np.clip(data, lo, hi)
    return (clipped - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Coast-wide reef probability inference over Algarve bbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/reef_habitat_mini/model/reef_cnn_best.pth",
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default="S2A_29SNB_20230901_0_L2A",
        help="Earth Search STAC item id (sentinel-2-l2a collection).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[-8.6, 36.85, -7.7, 37.30],
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="WGS-84 bounding box (lon_min lat_min lon_max lat_max).",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=128,
        help="Tile side length in pixels (must match model input size).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride between tile centres in pixels (smaller = denser).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for hotspot binary mask.",
    )
    parser.add_argument(
        "--top-hotspots",
        type=int,
        default=50,
        help="Number of top-probability hotspots to export to GeoJSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/reef_habitat_mini/predictions",
        help="Output directory for prob GeoTIFF, hotspots GeoJSON and log.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'cpu', 'cuda', or 'auto' (auto-detect).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=5000,
        help="Safety cap on total tiles to score (prevents full-coast OOM).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Raise ``ValueError`` on invalid CLI arguments."""
    lon_min, lat_min, lon_max, lat_max = args.bbox
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise ValueError(f"Invalid longitude range: {lon_min}..{lon_max}")
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(f"Invalid latitude range: {lat_min}..{lat_max}")
    if args.tile <= 0:
        raise ValueError(f"--tile must be > 0, got {args.tile}")
    if args.stride <= 0:
        raise ValueError(f"--stride must be > 0, got {args.stride}")
    if args.max_tiles <= 0:
        raise ValueError(f"--max-tiles must be > 0, got {args.max_tiles}")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")
    if args.top_hotspots < 0:
        raise ValueError(f"--top-hotspots must be >= 0, got {args.top_hotspots}")


# ─────────────────────────────────────────────────────────────────────────────
# STAC scene resolution
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_scene_assets(scene_id: str) -> dict:
    """
    Fetch a STAC item's asset URL mapping from Earth Search.

    Returns
    -------
    dict with keys: id, assets (alias→href), cloud_cover, datetime.

    Raises
    ------
    RuntimeError if the scene cannot be resolved.
    """
    url = f"{EARTH_SEARCH}/collections/{S2_COLLECTION}/items/{scene_id}"
    log.info("Resolving STAC scene: %s", url)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"STAC lookup failed: HTTP {resp.status_code} for {scene_id} "
            f"— body[:200]={resp.text[:200]!r}"
        )
    body = resp.json()
    return {
        "id":          body["id"],
        "cloud_cover": body.get("properties", {}).get("eo:cloud_cover", 99),
        "datetime":    body.get("properties", {}).get("datetime", ""),
        "assets":      {k: v["href"] for k, v in body["assets"].items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tile grid construction
# ─────────────────────────────────────────────────────────────────────────────

def compute_tile_grid(
    bbox: tuple[float, float, float, float],
    tile_px: int,
    stride_px: int,
    src_transform,
    src_width: int,
    src_height: int,
    src_crs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute tile origin pixel coordinates (row, col) that fit within the scene.

    Steps
    -----
    1. Convert the geographic bbox to the scene's pixel CRS if needed.
    2. Translate to pixel coords in the source raster.
    3. Slide a tile grid with ``stride_px`` over the bounding rectangle.
    4. Keep only tiles whose full footprint lies inside the raster.

    Returns
    -------
    rows, cols : 1-D arrays of equal length (tile origins in pixel space).
    """
    from rasterio.transform import rowcol

    lon_min, lat_min, lon_max, lat_max = bbox
    if src_crs and src_crs.to_epsg() != 4326:
        from pyproj import Transformer
        xformer = Transformer.from_crs("EPSG:4326", src_crs.to_epsg(), always_xy=True)
        x_min, y_min = xformer.transform(lon_min, lat_min)
        x_max, y_max = xformer.transform(lon_max, lat_max)
    else:
        x_min, y_min, x_max, y_max = lon_min, lat_min, lon_max, lat_max

    r0, c0 = rowcol(src_transform, x_min, y_max)   # top-left of bbox in pixel coords
    r1, c1 = rowcol(src_transform, x_max, y_min)   # bottom-right

    r0, r1 = sorted((int(r0), int(r1)))
    c0, c1 = sorted((int(c0), int(c1)))

    rows = np.arange(r0, r1 - tile_px + 1, stride_px, dtype=np.int64)
    cols = np.arange(c0, c1 - tile_px + 1, stride_px, dtype=np.int64)

    # Keep tiles fully inside raster.
    keep_r = (rows + tile_px) <= src_height
    keep_c = (cols + tile_px) <= src_width
    rows = rows[keep_r]
    cols = cols[keep_c]
    return rows, cols


def generate_tile_origins(
    rows: np.ndarray,
    cols: np.ndarray,
) -> list[tuple[int, int]]:
    """Cartesian product of row and column origins → list of (row, col)."""
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return list(zip(rr.ravel().tolist(), cc.ravel().tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Tile scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_tile(
    band_datasets: list,
    row: int,
    col: int,
    tile_px: int,
    model,
    device: str,
) -> Optional[float]:
    """
    Read a (4, tile_px, tile_px) window from each band, normalise, and predict.

    Returns probability in [0, 1] or None on failure (logged, never raised).
    """
    try:
        window = Window(col, row, tile_px, tile_px)
        channels: list[np.ndarray] = []
        for ds in band_datasets:
            arr = ds.read(1, window=window, boundless=False, fill_value=0)
            if arr.shape != (tile_px, tile_px):
                log.debug("Tile (%d,%d) has shape %s; skipping", row, col, arr.shape)
                return None
            channels.append(_normalise_band(arr.astype(np.float32), _band_name(ds)))

        patch = np.stack(channels, axis=0).astype(np.float32)  # (4, H, W)
        return predict_single(model, patch, device=device)
    except Exception as exc:
        log.warning("Tile (%d,%d) scoring failed: %s", row, col, exc)
        return None


def _band_name(ds) -> str:
    """Heuristic: infer band name from the rasterio dataset tags or URL."""
    tags = ds.tags() or {}
    for key in ("BAND", "band", "NAME", "name"):
        val = tags.get(key)
        if val:
            return val.upper()
    # Fallback: look at the filename inside the description.
    desc = ds.descriptions[0] if ds.descriptions else ""
    if desc.upper() in _BAND_ALIASES:
        return desc.upper()
    return "B02"  # default fallback — log only


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_prob_geotiff(
    out_path: Path,
    prob_grid: np.ndarray,
    transform,
    nodata: float = -9999.0,
) -> None:
    """Write the (n_rows, n_cols) tile-probability grid as a GeoTIFF."""
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "nodata":    nodata,
        "width":     prob_grid.shape[1],
        "height":    prob_grid.shape[0],
        "count":     1,
        "crs":       "EPSG:4326",
        "transform": transform,
        "compress":  "deflate",
        "predictor": 2,
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob_grid.astype(np.float32), 1)
    log.info("Wrote probability GeoTIFF: %s  shape=%s", out_path, prob_grid.shape)


def write_hotspots_geojson(
    out_path: Path,
    features: list[dict],
    threshold: float,
) -> None:
    """Write a GeoJSON FeatureCollection of top-K hotspots."""
    fc = {
        "type": "FeatureCollection",
        "metadata": {
            "threshold":   threshold,
            "n_features":  len(features),
            "score_field": "reef_probability",
        },
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)
    log.info("Wrote hotspots GeoJSON: %s  (n=%d)", out_path, len(features))


def write_scoring_log(
    out_path: Path,
    summary: dict,
) -> None:
    """Write a human-readable summary log."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Algarve tile scoring — summary\n")
        f.write("=" * 60 + "\n")
        for key, value in summary.items():
            f.write(f"  {key:<24s}: {value}\n")
        f.write("=" * 60 + "\n")
        if summary.get("top_hotspots"):
            f.write("\nTop-5 hotspots (lon, lat, prob):\n")
            for feat in summary["top_hotspots"][:5]:
                lon, lat = feat["geometry"]["coordinates"]
                f.write(
                    f"  ({lon:+.5f}, {lat:+.5f})  "
                    f"prob={feat['properties']['reef_probability']:.4f}\n"
                )
    log.info("Wrote scoring log: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(arg: str) -> str:
    """Resolve the --device flag to a concrete torch device string."""
    import torch
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main() -> int:
    args = parse_args()
    validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    bbox = tuple(args.bbox)

    log.info("Checkpoint : %s", args.checkpoint)
    log.info("Scene      : %s", args.scene_id)
    log.info("BBox       : %s", bbox)
    log.info("Tile/Stride: %d / %d px", args.tile, args.stride)
    log.info("Device     : %s", device)
    log.info("Out dir    : %s", out_dir)
    log.info("Max tiles  : %d", args.max_tiles)

    # 1. Resolve scene & band URLs
    scene = _fetch_scene_assets(args.scene_id)
    assets = scene["assets"]
    band_urls = []
    for band in BANDS:
        href = _band_url(assets, band)
        if href is None:
            raise RuntimeError(f"No asset for band {band} in scene {args.scene_id}")
        band_urls.append((band, href))
    log.info("Bands resolved: %s", [b for b, _ in band_urls])

    # 2. Load model (delegated to src.reef_classifier)
    t_load = time.time()
    model = load_model(args.checkpoint, device=device)
    log.info("Model loaded in %.1fs", time.time() - t_load)

    # 3. Open bands and compute tile grid
    band_datasets = []
    band_meta = None
    try:
        for band, href in band_urls:
            ds = rasterio.open(href)
            band_datasets.append(ds)
            if band_meta is None:
                band_meta = (ds.transform, ds.width, ds.height, ds.crs)

        transform, width, height, crs = band_meta
        rows, cols = compute_tile_grid(
            bbox=bbox,
            tile_px=args.tile,
            stride_px=args.stride,
            src_transform=transform,
            src_width=width,
            src_height=height,
            src_crs=crs,
        )
        log.info(
            "Scene %s: %d × %d px   →   grid origins rows=%d cols=%d",
            args.scene_id, width, height, len(rows), len(cols),
        )

        tile_origins = generate_tile_origins(rows, cols)
        n_total_grid = len(tile_origins)
        if n_total_grid == 0:
            log.error("No tiles fit inside bbox / scene; aborting.")
            return 2
        if n_total_grid > args.max_tiles:
            log.warning(
                "Grid has %d tiles, exceeding --max-tiles=%d; subsampling evenly.",
                n_total_grid, args.max_tiles,
            )
            idx = np.linspace(0, n_total_grid - 1, args.max_tiles).astype(int)
            tile_origins = [tile_origins[i] for i in idx]

        n_tiles = len(tile_origins)
        n_rows = len(rows)
        n_cols = len(cols)

        # 4. Score tiles; accumulate probabilities in a coarse grid keyed by
        #    (row_idx, col_idx).  Pixels outside scored tiles remain nodata.
        prob_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        scored = 0
        skipped = 0
        probs: list[float] = []
        tile_lonlat: list[tuple[float, float, float]] = []
        last_pct_log = -1

        t0 = time.time()
        for k, (r, c) in enumerate(tile_origins, start=1):
            p = score_tile(band_datasets, r, c, args.tile, model, device)
            # Map pixel origin back to its grid index.
            ri = int(np.searchsorted(rows, r))
            ci = int(np.searchsorted(cols, c))
            if p is None:
                skipped += 1
                continue
            prob_grid[ri, ci] = p
            probs.append(p)
            scored += 1
            # Geographic centre of tile (use the first band dataset).
            cx = transform.c + (c + args.tile / 2.0) * transform.a
            cy = transform.f + (r + args.tile / 2.0) * transform.e
            tile_lonlat.append((cx, cy, p))

            pct = int(100 * k / n_tiles)
            if pct // 10 != last_pct_log // 10:
                last_pct_log = pct
                log.info(
                    "Scored %d/%d tiles (%d%%)  mean_prob=%.3f  eta=%.0fs",
                    k, n_tiles, pct,
                    float(np.mean(probs)) if probs else 0.0,
                    (time.time() - t0) * (n_tiles - k) / max(k, 1),
                )
        elapsed = time.time() - t0
    finally:
        for ds in band_datasets:
            try:
                ds.close()
            except Exception:
                pass

    # 5. Top-K hotspots
    sorted_tiles = sorted(tile_lonlat, key=lambda t: t[2], reverse=True)
    top_k = sorted_tiles[: args.top_hotspots]
    features = []
    for lon, lat, prob in top_k:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "reef_probability": round(float(prob), 6),
                "scene_id":         args.scene_id,
                "tile_px":          args.tile,
                "stride_px":        args.stride,
            },
        })

    # 6. Outputs
    # Build a coarse output transform: one output pixel per tile origin.
    # Origin = top-left tile centre; pixel size = stride in the source raster's units.
    if len(rows) > 0 and len(cols) > 0:
        x0 = transform.c + (cols[0]) * transform.a
        y0 = transform.f + (rows[0]) * transform.e
        out_transform = from_bounds(
            x0, y0 + len(rows) * transform.e,
            x0 + len(cols) * transform.a, y0,
            len(cols), len(rows),
        )
    else:
        out_transform = transform

    # Replace NaNs with nodata for writing.
    write_grid = np.where(np.isnan(prob_grid), -9999.0, prob_grid).astype(np.float32)
    prob_tif = out_dir / "algarve_reef_prob.tif"
    write_prob_geotiff(prob_tif, write_grid, out_transform, nodata=-9999.0)

    hotspots_geojson = out_dir / "algarve_reef_hotspots.geojson"
    write_hotspots_geojson(hotspots_geojson, features, args.threshold)

    # 7. Summary
    above = int(np.sum(np.array(probs) >= args.threshold)) if probs else 0
    mean_p = float(np.mean(probs)) if probs else 0.0
    summary = {
        "checkpoint":       str(args.checkpoint),
        "scene_id":         args.scene_id,
        "bbox":             list(bbox),
        "tile_px":          args.tile,
        "stride_px":        args.stride,
        "threshold":        args.threshold,
        "max_tiles_cap":    args.max_tiles,
        "tiles_scored":     scored,
        "tiles_skipped":    skipped,
        "tiles_total_grid": n_total_grid,
        "tiles_above_thr":  above,
        "mean_probability": round(mean_p, 6),
        "elapsed_seconds":  round(elapsed, 2),
        "top_hotspots":     features,
        "device":           device,
    }
    log_path = out_dir / "scoring_log.txt"
    write_scoring_log(log_path, summary)

    # 8. Final stdout summary block
    log.info("=" * 60)
    log.info("Coast-wide scoring complete")
    log.info("  tiles scored       : %d", scored)
    log.info("  tiles skipped      : %d", skipped)
    log.info("  mean probability   : %.4f", mean_p)
    log.info("  tiles ≥ threshold  : %d  (threshold=%.2f)", above, args.threshold)
    log.info("  elapsed            : %.1fs", elapsed)
    log.info("  outputs            : %s", out_dir)
    log.info("  top-5 hotspots:")
    for feat in features[:5]:
        lon, lat = feat["geometry"]["coordinates"]
        log.info(
            "    (%+.5f, %+.5f)  prob=%.4f",
            lon, lat, feat["properties"]["reef_probability"],
        )
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())