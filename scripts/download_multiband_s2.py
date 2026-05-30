"""
download_multiband_s2.py
Download ALL Sentinel-2 L2A bands useful for reef/bathymetry detection
============================================================
Uses Microsoft Planetary Computer STAC (zero-auth) to:
  - Search for S2 L2A scene at a given point/date
  - Download 11 bands (B01-B12, B8A) clipped to AOI bbox
  - Resample 20m/60m bands to 10m grid (bilinear)
  - Save each band as GeoTIFF + metadata JSON

Usage:
  python download_multiband_s2.py --lat 37.068978 --lon -8.210328 --date 2024-10-15
  python download_multiband_s2.py --output-dir my_output --buffer-m 2000

Importable:
  from download_multiband_s2 import download_multiband
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from pystac_client import Client
import planetary_computer as pc

log = logging.getLogger("download_multiband_s2")

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

REEF_BANDS = {
    "B01": {"wl": "443nm", "desc": "Coastal Aerosol",    "res_m": 60},
    "B02": {"wl": "490nm", "desc": "Blue",               "res_m": 10},
    "B03": {"wl": "560nm", "desc": "Green",              "res_m": 10},
    "B04": {"wl": "665nm", "desc": "Red",                "res_m": 10},
    "B05": {"wl": "705nm", "desc": "Red Edge 1",         "res_m": 20},
    "B06": {"wl": "740nm", "desc": "Red Edge 2",         "res_m": 20},
    "B07": {"wl": "783nm", "desc": "Red Edge 3",         "res_m": 20},
    "B08": {"wl": "842nm", "desc": "NIR",                "res_m": 10},
    "B8A": {"wl": "865nm", "desc": "Narrow NIR",         "res_m": 20},
    "B11": {"wl": "1610nm", "desc": "SWIR1",             "res_m": 20},
    "B12": {"wl": "2190nm", "desc": "SWIR2",             "res_m": 20},
}

DEFAULT_LAT = 37.068978
DEFAULT_LON = -8.210328
DEFAULT_DATE = "2024-10-15"
DEFAULT_BUFFER_M = 1000
DEFAULT_OUTPUT_DIR = "reef_multiband_output"


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "download_multiband.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log.info("download_multiband_s2 started — log: %s", log_path)


def deg_offset(buffer_m: float) -> float:
    return buffer_m / 111_320.0


def read_cog_window(href: str, bbox_wgs84: tuple, target_shape: tuple = None,
                    target_transform: rasterio.Affine = None,
                    target_crs=None, resampling=Resampling.bilinear) -> tuple:
    """Read a COG window for a given bbox. Returns (data_2d, transform, crs, profile)."""
    west, south, east, north = bbox_wgs84
    with rasterio.open(href) as src:
        src_crs = src.crs
        read_west, read_south, read_east, read_north = west, south, east, north
        if src_crs and src_crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            read_west, read_south, read_east, read_north = transform_bounds(
                "EPSG:4326", src_crs, west, south, east, north
            )
        window = rasterio.windows.from_bounds(
            read_west, read_south, read_east, read_north, transform=src.transform
        )
        data = src.read(1, window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            driver="GTiff",
            compress="lzw",
            count=1,
        )
        return data, transform, src_crs, profile


def resample_to_10m(data: np.ndarray, src_transform: rasterio.Affine,
                    src_crs, src_dtype, bbox_wgs84: tuple,
                    target_res: float = 10.0,
                    resampling=Resampling.bilinear) -> tuple:
    """Resample a band array to a 10m target grid using bilinear interpolation.

    Returns (data_10m, target_transform, target_shape).
    """
    west, south, east, north = bbox_wgs84

    if src_crs and src_crs.to_epsg() == 32629:
        from rasterio.warp import transform_bounds
        west_t, south_t, east_t, north_t = transform_bounds(
            "EPSG:4326", src_crs, west, south, east, north
        )
        target_crs = src_crs
    else:
        target_crs = src_crs
        west_t, south_t, east_t, north_t = west, south, east, north
        if src_crs and src_crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            west_t, south_t, east_t, north_t = transform_bounds(
                "EPSG:4326", src_crs, west, south, east, north
            )

    width = max(1, int(round((east_t - west_t) / target_res)))
    height = max(1, int(round((north_t - south_t) / target_res)))
    target_transform = from_bounds(west_t, south_t, east_t, north_t, width, height)

    destination = np.zeros((height, width), dtype=np.float32)

    reproject(
        source=data.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=0,
        resampling=resampling,
    )

    return destination, target_transform, (height, width)


def download_multiband(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON,
                       date: str = DEFAULT_DATE, buffer_m: float = DEFAULT_BUFFER_M,
                       output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Download all reef-relevant S2 L2A bands to output_dir.

    Returns dict with keys: scene_id, bands (dict of band->filepath), meta_path.
    """
    os.makedirs(output_dir, exist_ok=True)

    d = deg_offset(buffer_m)
    bbox_wgs84 = (lon - d, lat - d, lon + d, lat + d)

    log.info("Searching Planetary Computer STAC for sentinel-2-l2a ...")
    log.info("  Point: (%.5f, %.5f), Date: %s, Buffer: %d m", lat, lon, date, buffer_m)

    catalog = Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
    time_range = f"{date}T00:00:00Z/{date}T23:59:59Z"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=time_range,
    )
    items = list(search.items())
    if not items:
        log.error("No S2 L2A scene found for %s at (%.5f, %.5f)", date, lat, lon)
        return {}

    item = items[0]
    scene_id = item.id
    log.info("Found scene: %s", scene_id)

    date_str = date.replace("-", "")
    band_files = {}
    band_info = {}

    for band_name, band_meta in REEF_BANDS.items():
        asset_key = band_name
        asset = item.assets.get(asset_key)
        if not asset:
            log.warning("  Asset %s not found in scene, skipping", band_name)
            continue

        out_path = os.path.join(output_dir, f"S2_{band_name}_{date_str}.tif")
        native_res = band_meta["res_m"]

        log.info("  Downloading %s (%s, %s, %d m) ...",
                 band_name, band_meta["desc"], band_meta["wl"], native_res)

        try:
            href = asset.href

            if native_res == 10:
                data, transform, crs, profile = read_cog_window(
                    href, bbox_wgs84
                )
            else:
                raw_data, raw_transform, raw_crs, raw_profile = read_cog_window(
                    href, bbox_wgs84
                )
                data, transform, (h, w) = resample_to_10m(
                    raw_data, raw_transform, raw_crs,
                    raw_profile.get("dtype", "uint16"),
                    bbox_wgs84, target_res=10.0
                )
                profile = raw_profile.copy()
                profile.update(height=h, width=w, transform=transform)

            profile.update(
                driver="GTiff",
                compress="lzw",
                count=1,
                dtype=data.dtype,
                nodata=0,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)

            file_size = os.path.getsize(out_path)
            band_files[band_name] = out_path
            band_info[band_name] = {
                "description": band_meta["desc"],
                "wavelength": band_meta["wl"],
                "native_resolution_m": native_res,
                "resampled_to_10m": native_res != 10,
                "file": out_path,
                "file_size_bytes": file_size,
                "shape": list(data.shape),
            }
            log.info("    ✓ %s saved → %s (%.1f KB, %dx%d)",
                     band_name, out_path, file_size / 1024,
                     data.shape[1], data.shape[0])

        except Exception as exc:
            log.error("    ✗ %s failed: %s", band_name, exc)

    meta = {
        "scene_id": scene_id,
        "date": date,
        "source": "planetary_computer_stac",
        "collection": "sentinel-2-l2a",
        "point": {"lat": lat, "lon": lon},
        "buffer_m": buffer_m,
        "aoi_bounds": {
            "west": bbox_wgs84[0],
            "south": bbox_wgs84[1],
            "east": bbox_wgs84[2],
            "north": bbox_wgs84[3],
        },
        "target_resolution_m": 10,
        "bands": band_info,
        "properties": {
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "s2_datatake_id": item.properties.get("s2:datatake_id"),
            "s2_mgrs_tile": item.properties.get("s2:mgrs_tile"),
            "s2_product_uri": item.properties.get("s2:product_uri"),
        },
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
    }

    meta_path = os.path.join(output_dir, f"S2_multiband_meta_{date_str}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata → %s", meta_path)

    log.info("=" * 60)
    log.info("Download summary:")
    log.info("  Scene: %s", scene_id)
    log.info("  Bands downloaded: %d / %d", len(band_files), len(REEF_BANDS))
    total_size = 0
    for bn, bi in band_info.items():
        sz_kb = bi["file_size_bytes"] / 1024
        total_size += bi["file_size_bytes"]
        resampled_tag = " [resampled→10m]" if bi["resampled_to_10m"] else ""
        log.info("    %s (%s) — %.1f KB — %dx%d%s",
                 bn, bi["description"], sz_kb,
                 bi["shape"][1], bi["shape"][0], resampled_tag)
    log.info("  Total size: %.1f KB (%.2f MB)", total_size / 1024, total_size / 1e6)
    log.info("  Output dir: %s", output_dir)

    return {
        "scene_id": scene_id,
        "bands": band_files,
        "meta_path": meta_path,
        "total_size_bytes": total_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download all Sentinel-2 L2A reef/bathymetry bands via Planetary Computer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT,
                        help="Latitude of center point")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON,
                        help="Longitude of center point")
    parser.add_argument("--date", default=DEFAULT_DATE,
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M,
                        help="Clip radius in meters")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for GeoTIFFs")

    args = parser.parse_args()

    setup_logging(args.output_dir)

    result = download_multiband(
        lat=args.lat,
        lon=args.lon,
        date=args.date,
        buffer_m=args.buffer_m,
        output_dir=args.output_dir,
    )

    if not result:
        log.error("Download failed — no results.")
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
