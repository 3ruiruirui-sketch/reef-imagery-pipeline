#!/usr/bin/env python3
"""
algarve_bathy_processor.py
==========================
Programmatic Python workflow to:
  1. Define a study area bbox (Faro -> Carvoeiro, Algarve, Portugal).
  2. Download European harmonized bathymetry from EMODnet WCS service.
  3. Reproject the bathymetric raster to PT-TM06 / ETRS89 (EPSG:3763).
  4. Clip and resample the grid to a standardized resolution (e.g., 50 meters).
  5. Export the clean output to a GeoTIFF.
"""

import os
import sys
import logging
import tempfile
import shutil
import requests
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("AlgarveBathyProcessor")

# ── Configuration & Spatial Bounds ───────────────────────────────────────────

# Study Area: Faro to Carvoeiro Bounding Box (WGS84, EPSG:4326)
# Coordinates cover immediate nearshore waters, reefs, and shoreline transitions.
STUDY_AREA_WGS84 = {
    "west": -8.50,   # West of Carvoeiro (~ -8.47)
    "south": 36.95,  # Offshore limit
    "east": -7.85,   # East of Faro (~ -7.93)
    "north": 37.15   # Shoreline/coastal land limit
}

# Coordinate Reference System (PT-TM06 / ETRS89, EPSG:3763)
TARGET_CRS_EPSG = 3763
TARGET_CRS = CRS.from_epsg(TARGET_CRS_EPSG)

# Target resolution in meters
TARGET_RESOLUTION_M = 50.0

# EMODnet Bathymetry WCS Endpoint
EMODNET_WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE_NAME = "emodnet:mean"

# Output configuration
OUTPUT_DIR = "outputs/bathy"
OUTPUT_FILENAME = "bathy_faro_carvoeiro_epsg3763_50m.tif"


# ── Step 1: Download EMODnet WCS ─────────────────────────────────────────────

def download_emodnet_wcs(bbox: dict, out_path: str) -> bool:
    """Download EMODnet mean bathymetry coverage for the bbox as a GeoTIFF."""
    log.info("Step 1: Requesting EMODnet WCS coverage...")
    
    # Calculate pixel dimensions for download grid
    # EMODnet base resolution is 1/128 degree (~115m). We query at a comparable scale.
    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    width_px = int(abs(e - w) * 111320.0 / 115.0)
    height_px = int(abs(n - s) * 111320.0 / 115.0)

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": COVERAGE_NAME,
        "CRS": "EPSG:4326",
        "BBOX": f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}",
        "WIDTH": str(width_px),
        "HEIGHT": str(height_px),
        "FORMAT": "image/tiff"
    }

    try:
        r = requests.get(EMODNET_WCS_URL, params=params, timeout=60)
        r.raise_for_status()
        
        # Verify content is not an XML error page
        content_type = r.headers.get("Content-Type", "")
        if "xml" in content_type.lower() or r.content[:5] in (b"<?xml", b"<Serv"):
            log.error("WCS service returned XML error message: %s", r.content[:300].decode("utf-8", errors="ignore"))
            return False
            
        # Safe-write pattern for virtiofs/FUSE mounts (writing to temp first)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".tif", dir=tempfile.gettempdir())
        try:
            os.close(temp_fd)
            with open(temp_path, "wb") as f:
                f.write(r.content)
            shutil.copy2(temp_path, out_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        log.info("Successfully downloaded raw WCS GeoTIFF to %s (%.1f kB)", 
                 out_path, os.path.getsize(out_path) / 1024)
        return True
    except Exception as exc:
        log.error("Failed to download EMODnet WCS bathymetry: %s", exc)
        return False


# ── Step 2: Reproject, Clip, and Resample ────────────────────────────────────

def process_raster_to_pt_tm06(src_path: str, dst_path: str, target_res: float) -> bool:
    """Reproject, clip, and resample the source raster to the Portuguese grid."""
    log.info("Step 2: Reprojecting and resampling to EPSG:%d (PT-TM06) @ %g m...", 
             TARGET_CRS_EPSG, target_res)

    try:
        with rasterio.open(src_path) as src:
            src_crs = src.crs or CRS.from_epsg(4326)
            
            # Calculate transform and dimensions for the target CRS
            transform, width, height = calculate_default_transform(
                src_crs, TARGET_CRS, src.width, src.height, *src.bounds
            )
            
            # Calculate target bounding box in EPSG:3763
            # We enforce exact metric resolution by creating a new affine transform matrix
            min_x, min_y, max_x, max_y = transform_bounds(src_crs, TARGET_CRS, *src.bounds)
            dst_width = int((max_x - min_x) / target_res)
            dst_height = int((max_y - min_y) / target_res)
            dst_transform = from_bounds(min_x, min_y, max_x, max_y, dst_width, dst_height)

            # Build metadata profile for the new GeoTIFF
            profile = src.profile.copy()
            profile.update({
                "crs": TARGET_CRS,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "dtype": rasterio.float32,
                "nodata": np.nan,
                "compress": "lzw"
            })

            # Read source band and allocate target array
            source_data = src.read(1).astype(np.float32)
            
            # Replace source nodata values with NaN for clean calculation
            if src.nodata is not None:
                source_data = np.where(source_data == src.nodata, np.nan, source_data)

            dest_data = np.empty((dst_height, dst_width), dtype=np.float32)
            dest_data.fill(np.nan)

            # Reproject raster array (resample using Bilinear interpolation for smooth depths)
            reproject(
                source=source_data,
                destination=dest_data,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear
            )

            # Write output using virtiofs-safe temp-copy pattern
            temp_fd, temp_path = tempfile.mkstemp(suffix=".tif", dir=tempfile.gettempdir())
            try:
                os.close(temp_fd)
                with rasterio.open(temp_path, "w", **profile) as dst:
                    dst.write(dest_data, 1)
                shutil.copy2(temp_path, dst_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            log.info("Processing complete. Saved aligned GeoTIFF to %s", dst_path)
            log.info("New dimensions: %d columns x %d rows. Resolution: %g m x %g m.", 
                     dst_width, dst_height, target_res, target_res)
            return True
            
    except Exception as exc:
        log.error("Failed processing raster transformation: %s", exc)
        return False


# ── Execution ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    raw_wcs_path = os.path.join(OUTPUT_DIR, "bathy_raw_wcs.tif")
    final_processed_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # 1. Download
    download_success = download_emodnet_wcs(STUDY_AREA_WGS84, raw_wcs_path)
    if not download_success:
        log.error("Workflow halted: Download failed.")
        sys.exit(1)
        
    # 2. Process
    process_success = process_raster_to_pt_tm06(
        raw_wcs_path, final_processed_path, TARGET_RESOLUTION_M
    )
    if not process_success:
        log.error("Workflow halted: Reprojection & resampling failed.")
        sys.exit(1)
        
    # 3. Clean up raw WCS download if desired
    if os.path.exists(raw_wcs_path):
        os.remove(raw_wcs_path)
        
    log.info("Algarve bathymetry processing workflow completed successfully!")


if __name__ == "__main__":
    main()
