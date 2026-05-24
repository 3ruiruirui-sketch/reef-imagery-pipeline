#!/usr/bin/env python3
"""
warm_ih_cache.py — Warm the IH/DGRM bathymetry cache for the full Algarve AOI
================================================================================

Crawls the entire Algarve coastal shelf and stores all isobath contours
locally so future queries are instant (no network dependency).

AOI: Portuguese Algarve coast from Sagres to Vila Real de Santo António
      (~36.9°N to 37.5°N, -8.9°W to -7.3°W)

Usage:
    python scripts/warm_ih_cache.py
    python scripts/warm_ih_cache.py --chunk-deg 0.15  # larger tiles (faster)

Author: 3ruiruirui-sketch
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ih_bathy_features import IHBathyDownloader, ALL_ISOBATHS

# ── Algarve AOI ───────────────────────────────────────────────────────────────
# Full Portuguese coast coverage (Caminha → Guadiana)
# Restricted to Algarve: from west of Sagres to Spanish border
ALGARVE_WEST = -8.90
ALGARVE_EAST = -7.30
ALGARVE_SOUTH = 36.85
ALGARVE_NORTH = 37.55

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Warm IH bathymetry cache for Algarve")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    parser.add_argument("--chunk-deg", type=float, default=0.10, help="Tile size in degrees")
    parser.add_argument("--timeout", type=int, default=45, help="Request timeout")
    parser.add_argument("--clear-first", action="store_true", help="Clear existing cache before warming")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    downloader = IHBathyDownloader(
        cache_dir=args.cache_dir,
        chunk_deg=args.chunk_deg,
        timeout=args.timeout,
    )

    if args.clear_first:
        log.info("Clearing existing cache...")
        downloader.clear_cache()

    log.info(
        "Warming IH cache for Algarve AOI: "
        "[%.2f, %.2f → %.2f, %.2f] (chunk=%.2f°)",
        ALGARVE_WEST, ALGARVE_SOUTH, ALGARVE_EAST, ALGARVE_NORTH, args.chunk_deg,
    )

    # Generate tiles to estimate work
    tiles = downloader._tile_bbox(
        ALGARVE_WEST, ALGARVE_SOUTH, ALGARVE_EAST, ALGARVE_NORTH, args.chunk_deg
    )
    log.info("Total tiles to fetch: %d", len(tiles))

    # Fetch all isobaths (not just reef subset) for maximum coverage
    features = downloader.fetch_for_aoi(
        min_lon=ALGARVE_WEST,
        min_lat=ALGARVE_SOUTH,
        max_lon=ALGARVE_EAST,
        max_lat=ALGARVE_NORTH,
        depths=ALL_ISOBATHS,
        use_cache=True,
    )

    # Summary
    depths_found = sorted({f["depth"] for f in features})
    total_length = sum(f.get("shape_leng", 0.0) for f in features)

    log.info("=" * 60)
    log.info("CACHE WARM COMPLETE")
    log.info("=" * 60)
    log.info("Total unique polylines: %d", len(features))
    log.info("Depths found: %s", depths_found)
    log.info("Total contour length: %.1f km", total_length / 1000.0)
    log.info("Cache directory: %s", Path(args.cache_dir).resolve())
    log.info("Next queries for any Algarve location will use cache.")
    log.info("=" * 60)

    # List cache files
    cache_files = sorted(Path(args.cache_dir).glob("ih_bathy_*.gpkg"))
    log.info("Cache files created: %d", len(cache_files))
    for f in cache_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        log.info("  %s (%.2f MB)", f.name, size_mb)

    return 0


if __name__ == "__main__":
    sys.exit(main())
