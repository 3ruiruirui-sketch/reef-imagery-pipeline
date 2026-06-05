#!/usr/bin/env python3
"""
ingest_lidar_dtm.py
Download DGT INCD LiDAR DTM tiles for SDB calibration.
API: https://dgt-be.a.incda.pt:8081 (public, no auth)
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.utils import write_band

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

API_BASE = "https://dgt-be.a.incda.pt:8081"
COLLECTION = "MDT-50cm"
OUTPUT_CRS = "EPSG:32629"


def bbox_from_point(lon: float, lat: float, buffer_m: float = 500.0) -> str:
    deg_per_m = 1.0 / 111320.0
    buf = buffer_m * deg_per_m
    return f"{lon - buf:.5f},{lat - buf:.5f},{lon + buf:.5f},{lat + buf:.5f}"


def fetch_tiles(bbox: str) -> list[dict]:
    url = f"{API_BASE}/collections/{COLLECTION}/items"
    params = {"bbox": bbox, "limit": 50, "f": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    fc = resp.json()
    features = fc.get("features", [])
    log.info("Found %d tiles for bbox %s", len(features), bbox)
    return features


def download_tiff(url: str, out_path: Path) -> Path | None:
    try:
        log.info("Downloading %s", url.split("/")[-1])
        tmp = out_path.with_suffix(".tmp.tif")
        with requests.get(url, timeout=120, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        shutil.move(str(tmp), str(out_path))
        log.info("Saved %.1f MB", out_path.stat().st_size / 1e6)
        return out_path
    except Exception as exc:
        log.warning("Failed: %s", exc)
        return None


def reproject_to_utm(src_path: Path, dst_path: Path, dst_crs: str = OUTPUT_CRS) -> Path:
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_meta = src.profile.copy()
        src_bounds = src.bounds
        src_crs_str = str(src.crs or "")

    if src_crs_str and src_crs_str != dst_crs:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs_str, dst_crs,
            src.width, src.height,
            *src_bounds,
        )
        dst_meta = src_meta.copy()
        dst_meta.update(
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            compress="lzw",
        )
        dst_data = np.zeros((dst_height, dst_width), dtype=src_data.dtype)
        reproject(
            source=src_data.astype(float),
            destination=dst_data,
            src_transform=src_meta["transform"],
            src_crs=src_crs_str,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        dst_data = np.where(dst_data == src.nodata, np.nan, dst_data)
        write_band(str(dst_path), dst_data, dst_meta, nodata=np.nan)
    else:
        shutil.copy(src_path, dst_path)
    return dst_path


def main():
    parser = argparse.ArgumentParser(description="Download DGT INCD LiDAR DTM tiles")
    parser.add_argument("--lon", type=float, help="Centre longitude (WGS84)")
    parser.add_argument("--lat", type=float, help="Centre latitude (WGS84)")
    parser.add_argument("--buffer-m", type=float, default=500.0)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    parser.add_argument("--output", default="data/lidar_tiles")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        bbox_str = ",".join(map(str, args.bbox))
    elif args.lon is not None and args.lat is not None:
        bbox_str = bbox_from_point(args.lon, args.lat, args.buffer_m)
    else:
        log.error("Provide --lon/--lat or --bbox MINX MINY MAXX MAXY")
        sys.exit(1)

    log.info("Bounding box: %s", bbox_str)
    features = fetch_tiles(bbox_str)

    if args.dry_run:
        for f in features:
            props = f.get("properties", {})
            href = f.get("assets", {}).get("Data", {}).get("href", "")
            log.info("  DRY %s  %.1f MB", href.split("/")[-1], props.get("file:size", 0) / 1e6)
        return

    downloaded = []
    for f in features:
        tile_id = f.get("id", "?")
        href = f.get("assets", {}).get("Data", {}).get("href", "")
        if not href:
            continue
        out_path = out_dir / f"{tile_id}_DTM.tif"
        result = download_tiff(href, out_path)
        if result:
            downloaded.append(result)

    log.info("Downloaded %d/%d tiles", len(downloaded), len(features))
    if not downloaded:
        log.error("No tiles downloaded")
        sys.exit(1)

    # Reproject first tile to check CRS
    sample = downloaded[0]
    reprojected = out_dir / "sample_reprojected.tif"
    reproject_to_utm(sample, reprojected)
    log.info("Sample reprojected to %s", reprojected)

    log.info("Done. Tiles in %s", out_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
