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

def process_target_date():
    target_dir = os.path.join(_PROJECT_ROOT, "reef_Output_Master", "reef_output_target_12m")
    ref_b02 = os.path.join(target_dir, "S2_B02_20250803.tif")
    
    if not os.path.exists(ref_b02):
        print(f"❌ Reference file not found: {ref_b02}")
        return
        
    # Read reference profile
    print("[1] Reading reference GeoTIFF profile...")
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
    
    print(f"    Target Area Bounds (WGS84): [{south}, {west}] to [{north}, {east}]")
    
    # Search STAC for 2026-02-22
    target_date = "2026-02-22"
    print(f"\n[2] Searching Planetary Computer STAC for Sentinel-2 scene on {target_date}...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    time_range = f"{target_date}T00:00:00Z/{target_date}T23:59:59Z"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [center_lon, center_lat]},
        datetime=time_range,
    )
    items = list(search.items())
    if not items:
        print(f"❌ No scene found for {target_date}")
        return
        
    item = items[0]
    print(f"    Found scene: {item.id}")
    
    # Download B02 and B03 windows
    bands_data = {}
    print(f"\n[3] Downloading B02 and B03 bands matching target bounds...")
    for band in ["B02", "B03"]:
        asset = item.assets.get(band)
        if not asset:
            print(f"❌ Asset {band} missing in STAC item!")
            return
            
        print(f"    Downloading {band} via COG window...")
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(asset.href) as src:
                src_crs = src.crs
                # Project bounding box to COG source CRS
                p_west, p_south, p_east, p_north = bounds
                if crs != src_crs:
                    p_west, p_south, p_east, p_north = transform_bounds(crs, src_crs, bounds[0], bounds[1], bounds[2], bounds[3])
                
                window = rasterio.windows.from_bounds(p_west, p_south, p_east, p_north, transform=src.transform)
                # Read band data and resample to reference size
                data = src.read(1, window=window, out_shape=(height, width))
                bands_data[band] = data.astype(np.float32)
                
        # Write band GeoTIFF
        out_tif = os.path.join(target_dir, f"S2_{band}_20260222.tif")
        profile.update(dtype=rasterio.uint16 if band in ["B02", "B03"] else rasterio.float32)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(data.astype(np.uint16), 1)
        print(f"    ✓ Saved: {out_tif}")

    # Compute ratio
    print(f"\n[4] Computing log(B02)/log(B03) ratio...")
    b02 = bands_data["B02"]
    b03 = bands_data["B03"]
    
    # Handle zeros and NaNs
    b02_clean = np.where(b02 > 0, b02, np.nan)
    b03_clean = np.where(b03 > 0, b03, np.nan)
    ratio = np.log(b02_clean) / np.log(b03_clean)
    
    ratio_tif = os.path.join(target_dir, "ratio_B02_B03_20260222.tif")
    ratio_profile = profile.copy()
    ratio_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(ratio_tif, "w", **ratio_profile) as dst:
        dst.write(ratio.astype(np.float32), 1)
    print(f"    ✓ Saved: {ratio_tif}")

    # Generate Leaflet-compatible styled PNG overlays (with transparent backgrounds for zero/nan)
    print(f"\n[5] Generating styled PNG tiles for Leaflet dashboard...")
    tiles_dir = os.path.join(_PROJECT_ROOT, "dashboard", "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    def save_transparent_png(arr, colormap, out_path, is_ratio=False):
        valid = arr[~np.isnan(arr) & (arr > 0)]
        if valid.size == 0:
            print(f"    ⚠️ Array contains no valid data for {out_path}")
            return
            
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
        
        # Apply colormap
        cmap = matplotlib.colormaps.get_cmap(colormap)
        rgba = cmap(norm)  # float RGBA [0.0, 1.0]
        
        # Make nodata / zero pixels transparent
        nodata_mask = np.isnan(arr) | (arr <= 0)
        rgba[nodata_mask, 3] = 0.0  # Set alpha to 0
        
        # Save image
        plt.imsave(out_path, rgba)
        print(f"    ✓ Saved tile overlay: {out_path}")

    save_transparent_png(ratio, "viridis", os.path.join(tiles_dir, "reef_output_target_12m_20260222.png"), is_ratio=True)
    save_transparent_png(b02, "Blues_r", os.path.join(tiles_dir, "reef_output_target_12m_20260222_b02.png"))
    save_transparent_png(b03, "Greens_r", os.path.join(tiles_dir, "reef_output_target_12m_20260222_b03.png"))

    # Write S2 Meta JSON
    meta_json = os.path.join(target_dir, "S2_meta_20260222.json")
    meta = {
        "scene": item.id,
        "id": item.properties.get("s2:granule_id", ""),
        "date": "2026-02-22",
        "source": "pc"
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    ✓ Saved metadata: {meta_json}")

    # Register in layers_meta.json
    print(f"\n[6] Registering new date in dashboard/layers_meta.json...")
    layers_meta_path = os.path.join(_PROJECT_ROOT, "dashboard", "layers_meta.json")
    
    with open(layers_meta_path, "r") as f:
        layers = json.load(f)
        
    # Check if entry already exists
    exists = False
    for l in layers:
        if l.get("dir") == "reef_output_target_12m" and l.get("date") == "20260222":
            exists = True
            break
            
    if not exists:
        # Get bounds from reference entry or compute from bounds coordinates
        # Bounding coordinates of EPSG:3763 coordinates back to WGS84 bounds
        wgs_bounds = [
            [south, west],
            [north, east]
        ]
        
        new_entry = {
            "dir": "reef_output_target_12m",
            "ratio_tif": "reef_output_target_12m/ratio_B02_B03_20260222.tif",
            "date": "20260222",
            "bounds": wgs_bounds,
            "width": width,
            "height": height,
            "tile_png": "tiles/reef_output_target_12m_20260222.png",
            "tile_b02": "tiles/reef_output_target_12m_20260222_b02.png",
            "tile_b03": "tiles/reef_output_target_12m_20260222_b03.png"
        }
        
        # Insert after other reef_output_target_12m entries or append
        inserted = False
        for idx, l in enumerate(layers):
            if l.get("dir") == "reef_output_target_12m" and l.get("date") > "20260222":
                layers.insert(idx, new_entry)
                inserted = True
                break
        if not inserted:
            layers.append(new_entry)
            
        with open(layers_meta_path, "w") as f:
            json.dump(layers, f, indent=2)
        print(f"    ✓ Registered in {layers_meta_path} successfully!")
    else:
        print(f"    ℹ Entry already registered in layers_meta.json.")
        
    print("\n🎉 Done! All tasks completed successfully.")

if __name__ == "__main__":
    process_target_date()
