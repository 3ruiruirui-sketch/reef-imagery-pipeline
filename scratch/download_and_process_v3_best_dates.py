#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import planetary_computer as pc
from pystac_client import Client
from datetime import datetime

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def process_v3_dates():
    v3_dir = os.path.join(_PROJECT_ROOT, "reef_Output_Master", "reef_output_v3")
    ref_b02 = os.path.join(v3_dir, "S2_B02_20250925.tif")
    
    if not os.path.exists(ref_b02):
        print(f"❌ Reference file not found: {ref_b02}")
        return
        
    print("[1] Reading reference GeoTIFF profile from reef_output_v3...")
    with rasterio.open(ref_b02) as ref:
        profile = ref.profile.copy()
        bounds = ref.bounds
        crs = ref.crs
        width = ref.width
        height = ref.height
        transform = ref.transform
        
    # Project bounds to WGS-84 (EPSG:4326) to search STAC
    west, south, east, north = bounds
    if crs and crs.to_epsg() != 4326:
        west, south, east, north = transform_bounds(crs, "EPSG:4326", west, south, east, north)
        
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2
    
    dates = ["2026-03-31", "2023-02-03"]
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    layers_meta_entries = []
    
    for date in dates:
        print(f"\n==========================================")
        print(f"🔄 Processing date: {date}")
        print(f"==========================================")
        
        time_range = f"{date}T00:00:00Z/{date}T23:59:59Z"
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [center_lon, center_lat]},
            datetime=time_range,
        )
        items = list(search.items())
        if not items:
            print(f"❌ No scene found for {date}")
            continue
            
        item = items[0]
        print(f"✓ Found scene: {item.id}")
        
        bands_data = {}
        
        for band in ["B02", "B03"]:
            asset = item.assets.get(band)
            if not asset:
                print(f"❌ Asset {band} missing in STAC item!")
                continue
                
            print(f"    Downloading {band} via COG window...")
            with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(asset.href) as src:
                    src_crs = src.crs
                    p_west, p_south, p_east, p_north = bounds
                    if crs != src_crs:
                        p_west, p_south, p_east, p_north = transform_bounds(crs, src_crs, bounds[0], bounds[1], bounds[2], bounds[3])
                    
                    window = rasterio.windows.from_bounds(p_west, p_south, p_east, p_north, transform=src.transform)
                    data = src.read(1, window=window, out_shape=(height, width))
                    bands_data[band] = data.astype(np.float32)
                    
        # Save raw GeoTIFF bands
        date_str = date.replace("-", "")
        for band in ["B02", "B03"]:
            out_tif = os.path.join(v3_dir, f"S2_{band}_{date_str}.tif")
            profile.update(dtype=rasterio.uint16)
            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(bands_data[band].astype(np.uint16), 1)
            print(f"    ✓ Saved: {out_tif}")

        # Compute ratio
        b02 = bands_data["B02"]
        b03 = bands_data["B03"]
        b02_clean = np.where(b02 > 0, b02, np.nan)
        b03_clean = np.where(b03 > 0, b03, np.nan)
        ratio = np.log(b02_clean) / np.log(b03_clean)
        
        ratio_tif = os.path.join(v3_dir, f"ratio_B02_B03_{date_str}.tif")
        ratio_profile = profile.copy()
        ratio_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(ratio_tif, "w", **ratio_profile) as dst:
            dst.write(ratio.astype(np.float32), 1)
        print(f"    ✓ Saved ratio GeoTIFF: {ratio_tif}")

        # Save transparent Leaflet PNG tiles
        tiles_dir = os.path.join(_PROJECT_ROOT, "dashboard", "tiles")
        os.makedirs(tiles_dir, exist_ok=True)
        
        def save_transparent_png(arr, colormap, out_path):
            valid = arr[~np.isnan(arr) & (arr > 0)]
            if valid.size == 0:
                return
            vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
            norm = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
            cmap = matplotlib.colormaps.get_cmap(colormap)
            rgba = cmap(norm)
            nodata_mask = np.isnan(arr) | (arr <= 0)
            rgba[nodata_mask, 3] = 0.0  # Set transparency
            plt.imsave(out_path, rgba)
            print(f"    ✓ Saved tile overlay: {out_path}")

        tile_png = f"tiles/reef_output_v3_{date_str}.png"
        tile_b02 = f"tiles/reef_output_v3_{date_str}_b02.png"
        tile_b03 = f"tiles/reef_output_v3_{date_str}_b03.png"
        
        save_transparent_png(ratio, "viridis", os.path.join(_PROJECT_ROOT, "dashboard", tile_png))
        save_transparent_png(b02, "Blues_r", os.path.join(_PROJECT_ROOT, "dashboard", tile_b02))
        save_transparent_png(b03, "Greens_r", os.path.join(_PROJECT_ROOT, "dashboard", tile_b03))

        # Generate visual analysis side-by-side PNG
        plot_path = os.path.join(v3_dir, f"ratio_analysis_{date_str}.png")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        b02_min, b02_max = np.nanpercentile(b02_clean, 2), np.nanpercentile(b02_clean, 98)
        b03_min, b03_max = np.nanpercentile(b03_clean, 2), np.nanpercentile(b03_clean, 98)
        r_min, r_max = np.nanpercentile(ratio, 2), np.nanpercentile(ratio, 98)
        
        axes[0].imshow(b02_clean, cmap="Blues_r", vmin=b02_min, vmax=b02_max);  axes[0].set_title("B02 (Blue)")
        axes[1].imshow(b03_clean, cmap="Greens_r", vmin=b03_min, vmax=b03_max); axes[1].set_title("B03 (Green)")
        im = axes[2].imshow(ratio, cmap="RdYlBu_r", vmin=r_min, vmax=r_max)
        axes[2].set_title("log(B02)/log(B03) Bathymetry")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Original Analysis Area — Sentinel-2 Dynamic Analysis  {date}", fontsize=13)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        
        # Copy to tiles directory for embedding
        tiles_plot_path = os.path.join(tiles_dir, f"analysis_v3_{date_str}.png")
        plt.savefig(tiles_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Saved side-by-side analysis: {plot_path}")

        # Construct bounds array in format [[south, west], [north, east]]
        wgs_bounds = [
            [south, west],
            [north, east]
        ]
        
        layers_meta_entries.append({
            "dir": "reef_output_v3",
            "ratio_tif": f"reef_output_v3/ratio_B02_B03_{date_str}.tif",
            "date": date_str,
            "bounds": wgs_bounds,
            "width": width,
            "height": height,
            "tile_png": tile_png,
            "tile_b02": tile_b02,
            "tile_b03": tile_b03
        })

    # Register in layers_meta.json
    print(f"\n[6] Registering new entries in dashboard/layers_meta.json...")
    layers_meta_path = os.path.join(_PROJECT_ROOT, "dashboard", "layers_meta.json")
    with open(layers_meta_path, "r") as f:
        layers = json.load(f)
        
    for entry in layers_meta_entries:
        # Check if already registered
        exists = False
        for l in layers:
            if l.get("dir") == entry["dir"] and l.get("date") == entry["date"]:
                exists = True
                break
        if not exists:
            # Add to the beginning after the first key or sort it
            layers.append(entry)
            print(f"    Added entry for date: {entry['date']}")
            
    with open(layers_meta_path, "w") as f:
        json.dump(layers, f, indent=2)
    print(f"✓ Registered in layers_meta.json successfully!")
    print("\n🎉 Success! Original Analysis Area target date processing fully complete.")

if __name__ == "__main__":
    process_v3_dates()
