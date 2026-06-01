#!/usr/bin/env python3
"""
generate_specific_dates_1200dpi.py
==================================
Generates high-resolution (1200 DPI) enhanced B02 "blue images" (using reef_cmap)
specifically for target dates 2025-09-25 and 2025-10-05 at both primary dive sites:
  1. Pedra do Alto (CORRECTED)
  2. Pedra de Santa Eulália
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
from skimage.restoration import denoise_nl_means, estimate_sigma

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DIVE_SITES = [
    ("pedra_do_alto_corrected", 37.05815,  -8.20982,  16),
    ("pedra_santa_eulalia",     37.068978, -8.210328, 12),
]

TARGET_DATES = ["2025-09-25", "2023-09-01"]

OUT_DIR = Path("../outputs/dive_sites_1200dpi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Blue reef colormap
reef_cmap = LinearSegmentedColormap.from_list(
    "reef", ["#000022", "#001155", "#003388", "#0066aa", "#00aacc", "#44ddaa", "#aaffaa"]
)

def find_scene_for_date(lat, lon, date_str):
    """Find the scene for a specific date and location."""
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    target = pd.to_datetime(date_str).date()
    next_day = target + pd.Timedelta(days=1)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{target.isoformat()}/{next_day.isoformat()}",
        query={"eo:cloud_cover": {"lt": 70}}, # Allow higher STAC cloud cover (local check is key)
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))

def download_and_enhance(item, lat, lon, buffer_m=500):
    """Download B02 and enhance it."""
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    try:
        with env:
            with rasterio.open(item.assets["B02"].href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                
                min_x = x - buffer_m
                min_y = y - buffer_m
                max_x = x + buffer_m
                max_y = y + buffer_m
                
                inv_transform = ~src.transform
                col_min, row_max = inv_transform * (min_x, min_y)
                col_max, row_min = inv_transform * (max_x, max_y)
                
                window = Window(
                    col_off=int(col_min),
                    row_off=int(row_min),
                    width=int(col_max - col_min),
                    height=int(row_max - row_min)
                )
                
                b02_arr = src.read(1, window=window).astype(np.float32)
    except Exception as e:
        print(f"      Error downloading B02: {e}")
        return None

    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    if b02.max() == 0 or np.all(np.isnan(b02)):
        return None

    # Enhance B02
    p95 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
    b02_corr = np.clip(b02 - 0.8 * p95 * 0.05, 0, 1.0)
    
    sigma_est = np.mean(estimate_sigma(b02_corr))
    b02_denoised = denoise_nl_means(b02_corr, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    
    b02_16 = np.clip(b02_denoised * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(4, 4))
    b02_clahe = clahe.apply(b02_16).astype(np.float32) / 65535.0
    
    b02_enhanced = b02_denoised * 0.5 + b02_clahe * 0.5
    return b02_enhanced

def main():
    print("=" * 80)
    print("  HIGH-RESOLUTION (1200 DPI) IMAGES GENERATION FOR SPECIFIC TARGET DATES")
    print(f"  Target dates: {TARGET_DATES}")
    print(f"  Output directory: {OUT_DIR.resolve()}")
    print("=" * 80)

    for idx, (name, lat, lon, depth) in enumerate(DIVE_SITES):
        print(f"\n[{idx+1}/{len(DIVE_SITES)}] Processing dive site: {name}...")
        
        for date_str in TARGET_DATES:
            print(f"  -> Targeting date: {date_str}...")
            item = find_scene_for_date(lat, lon, date_str)
            if not item:
                print(f"     SKIPPING: No scene found for date {date_str}")
                continue
                
            print(f"     Scene found (STAC cloud: {item.properties.get('eo:cloud_cover'):.1f}%)")
            
            b02_enhanced = download_and_enhance(item, lat, lon)
            if b02_enhanced is None:
                print("     SKIPPING: Download or enhancement failed.")
                continue
                
            # Stretch limits
            p2, p98 = np.percentile(b02_enhanced[b02_enhanced > 0], [2, 98]) if np.any(b02_enhanced > 0) else (0, 1)
            img_stretch = np.clip((b02_enhanced - p2) / (p98 - p2 + 1e-12), 0, 1)

            # Plot and save at 1200 DPI
            fig, ax = plt.subplots(figsize=(6, 6), facecolor="#000022")
            ax.imshow(img_stretch, cmap=reef_cmap, interpolation="bilinear")
            ax.axis("off")
            
            # Borderless save
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            out_path = OUT_DIR / f"{name}_{date_str}_blue_1200dpi.png"
            fig.savefig(out_path, dpi=1200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            
            print(f"     SUCCESS: Saved high-res image to: {out_path.name}")

    print("\n" + "=" * 80)
    print(f"  COMPLETE! All target date high-res images saved to: {OUT_DIR.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
