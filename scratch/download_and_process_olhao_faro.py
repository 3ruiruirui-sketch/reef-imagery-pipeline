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
from pyproj import Transformer

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.reef_ml_predictor_acolite import make_snr_map

def download_and_process():
    lat = 37.036578
    lon = -8.074393
    buffer_m = 2000.0  # 4km x 4km area
    
    # Bounding box in WGS-84
    # 1 degree lat ≈ 111,000m; 1 degree lon ≈ 89,000m at 37°N
    d_lat = buffer_m / 111000.0
    d_lon = buffer_m / 89000.0
    west = lon - d_lon
    east = lon + d_lon
    south = lat - d_lat
    north = lat + d_lat
    
    output_dir = os.path.join(_PROJECT_ROOT, "reef_Output_Master", "reef_output_olhao_faro")
    os.makedirs(output_dir, exist_ok=True)
    
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
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=time_range,
        )
        items = list(search.items())
        if not items:
            print(f"❌ No scene found for {date}")
            continue
            
        item = items[0]
        print(f"✓ Found scene: {item.id}")
        
        bands_data = {}
        out_profile = None
        
        for band in ["B02", "B03"]:
            asset = item.assets.get(band)
            if not asset:
                print(f"❌ Asset {band} missing in STAC item!")
                continue
                
            print(f"    Downloading {band} via COG window...")
            with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(asset.href) as src:
                    src_crs = src.crs
                    
                    # Project WGS-84 bounds to COG source CRS
                    p_west, p_south, p_east, p_north = west, south, east, north
                    if src_crs.to_epsg() != 4326:
                        p_west, p_south, p_east, p_north = transform_bounds("EPSG:4326", src_crs, west, south, east, north)
                        
                    window = rasterio.windows.from_bounds(p_west, p_south, p_east, p_north, transform=src.transform)
                    
                    data = src.read(1, window=window)
                    bands_data[band] = data.astype(np.float32)
                    
                    # Setup output profile
                    if out_profile is None:
                        out_profile = src.profile.copy()
                        out_profile.update(
                            height=data.shape[0],
                            width=data.shape[1],
                            transform=src.window_transform(window),
                            driver="GTiff",
                            compress="lzw"
                        )
            
            # Save raw GeoTIFF band
            date_str = date.replace("-", "")
            out_tif = os.path.join(output_dir, f"S2_{band}_{date_str}.tif")
            out_profile.update(dtype=rasterio.uint16)
            with rasterio.open(out_tif, "w", **out_profile) as dst:
                dst.write(data.astype(np.uint16), 1)
            print(f"    ✓ Saved: {out_tif}")

        # Compute ratio
        b02 = bands_data["B02"]
        b03 = bands_data["B03"]
        b02_clean = np.where(b02 > 0, b02, np.nan)
        b03_clean = np.where(b03 > 0, b03, np.nan)
        ratio = np.log(b02_clean) / np.log(b03_clean)
        
        ratio_tif = os.path.join(output_dir, f"ratio_B02_B03_{date_str}.tif")
        ratio_profile = out_profile.copy()
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

        tile_png = f"tiles/reef_output_olhao_faro_{date_str}.png"
        tile_b02 = f"tiles/reef_output_olhao_faro_{date_str}_b02.png"
        tile_b03 = f"tiles/reef_output_olhao_faro_{date_str}_b03.png"
        
        save_transparent_png(ratio, "viridis", os.path.join(_PROJECT_ROOT, "dashboard", tile_png))
        save_transparent_png(b02, "Blues_r", os.path.join(_PROJECT_ROOT, "dashboard", tile_b02))
        save_transparent_png(b03, "Greens_r", os.path.join(_PROJECT_ROOT, "dashboard", tile_b03))

        # Generate visual analysis side-by-side PNG and save under outputs / and dashboard / tiles for embedding
        plot_path = os.path.join(output_dir, f"ratio_analysis_{date_str}.png")
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
        fig.suptitle(f"Faro-Olhão Reef — Sentinel-2 Dynamic Analysis  {date}", fontsize=13)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        # Also copy this plot to tiles dir so we can embed it
        tiles_plot_path = os.path.join(tiles_dir, f"analysis_{date_str}.png")
        plt.savefig(tiles_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Saved side-by-side analysis: {plot_path}")

        # Construct bounds array in format [[south, west], [north, east]]
        wgs_bounds = [
            [south, west],
            [north, east]
        ]
        
        layers_meta_entries.append({
            "dir": "reef_output_olhao_faro",
            "ratio_tif": f"reef_output_olhao_faro/ratio_B02_B03_{date_str}.tif",
            "date": date_str,
            "bounds": wgs_bounds,
            "width": out_profile["width"],
            "height": out_profile["height"],
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
            layers.append(entry)
            print(f"    Added entry for date: {entry['date']}")
            
    with open(layers_meta_path, "w") as f:
        json.dump(layers, f, indent=2)
    print(f"✓ Registered in layers_meta.json successfully!")
    
    # Also update index.html to have a nice name map for this folder
    print(f"\n[7] Updating dashboard name mapping...")
    index_path = os.path.join(_PROJECT_ROOT, "dashboard", "index.html")
    with open(index_path, "r") as f:
        html = f.read()
        
    target_str = "'target east reef': 'Target (East Reef)',"
    replacement_str = "'target east reef': 'Target (East Reef)',\n          'olhao faro': 'Faro-Olhão Reef Area',"
    if "olhao faro" not in html:
        html = html.replace(target_str, replacement_str)
        with open(index_path, "w") as f:
            f.write(html)
        print("    ✓ Added 'Faro-Olhão Reef Area' mapping in index.html")
        
    print("\n🎉 Success! Faro-Olhão target date processing fully complete.")

if __name__ == "__main__":
    download_and_process()
