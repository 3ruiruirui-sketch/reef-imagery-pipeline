#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import requests
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import transform_bounds
from scipy.ndimage import uniform_filter, generic_filter, label as ndimage_label

# Set up project root on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from shapely.geometry import shape
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("geopandas/shapely missing!")

def export_regional_data():
    scratch_dir = os.path.dirname(os.path.abspath(__file__))
    bathy_tif = os.path.join(scratch_dir, "regional_emodnet.tif")
    dashboard_dir = os.path.join(_PROJECT_ROOT, "dashboard")
    tiles_dir = os.path.join(dashboard_dir, "tiles")
    
    if not os.path.exists(bathy_tif):
        print("Error: Regional tif not found. Run scratch/check_regional_bathy.py first!")
        return
        
    with rasterio.open(bathy_tif) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        crs = src.crs
        
        # Calculate WGS84 bounds
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326",
            src.bounds.left, src.bounds.bottom,
            src.bounds.right, src.bounds.top
        )
        bounds_wgs84 = [[south, west], [north, east]]
        print(f"Regional Bounds: {bounds_wgs84}")

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # 1. Generate beautiful colored PNG tile (Viridis colormap)
    print("Generating colored bathymetry tile...")
    valid = dem[np.isfinite(dem)]
    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    
    # Normalize
    normalized = np.clip((dem - vmin) / (vmax - vmin + 1e-10), 0, 1)
    
    # Apply colormap
    cmap = plt.get_cmap("viridis")
    colored = cmap(normalized)  # RGBA
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save as PNG
    png_path = os.path.join(tiles_dir, "regional_bathymetry.png")
    plt.imsave(png_path, colored_rgb)
    print(f"✓ Saved colored tile to {png_path}")
    
    # 2. Vectorise regional mounds
    if not HAS_GEO:
        print("Cannot vectorise candidates due to missing geopandas/shapely")
        return
        
    print("Vectorising benthic mounds...")
    depth_mask = (dem >= -40.0) & (dem <= -10.0) & np.isfinite(dem)
    
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dx_m = res_x * 111320.0 * np.cos(np.radians(37.05))
    dy_m = res_y * 111320.0
    
    dem_filled = np.where(np.isfinite(dem), dem, 0.0)
    
    # TRI (rugosity)
    def _tri_kernel(x):
        centre = x[len(x) // 2]
        diffs = x - centre
        diffs[len(x) // 2] = 0.0
        return float(np.sqrt(np.sum(diffs**2) / 8.0))
        
    tri = generic_filter(dem_filled, _tri_kernel, size=3, mode="nearest")
    tri = np.where(np.isfinite(dem), tri, np.nan)
    
    # BPI broad
    mean_broad = uniform_filter(dem_filled, size=15, mode="nearest")
    bpi = dem - mean_broad
    bpi = np.where(np.isfinite(dem), bpi, np.nan)
    
    tri_valid = tri[depth_mask & np.isfinite(tri)]
    tri_thresh = np.percentile(tri_valid, 75) if len(tri_valid) > 0 else 1.0
    
    bpi_valid = bpi[depth_mask & np.isfinite(bpi)]
    bpi_thresh = np.percentile(bpi_valid, 75) if len(bpi_valid) > 0 else 0.2
    
    candidates = depth_mask & (tri >= tri_thresh) & (bpi >= bpi_thresh)
    labelled, n_features = ndimage_label(candidates)
    
    # Filter small components
    sizes = np.bincount(labelled.ravel())
    filtered = np.where(np.isin(labelled, np.where(sizes < 2)[0]), 0, candidates.astype(np.uint8))
    
    # Vectorise
    geoms = []
    properties = []
    for idx, (geom_dict, val) in enumerate(rasterio_shapes(filtered, mask=filtered, transform=transform)):
        if val != 1:
            continue
        geom = shape(geom_dict)
        geoms.append(geom)
        
        # Calculate stats for this polygon
        mask_poly = (labelled == labelled[int(np.mean(np.where(filtered == 1)[0])), int(np.mean(np.where(filtered == 1)[1]))]) # approximate center
        
        # Get coordinates of center
        poly_center = geom.centroid
        lon_c, lat_c = poly_center.x, poly_center.y
        
        # Simple stats
        area_m2 = geom.area * (111320.0**2)
        
        # Estimate depth, tri, bpi at this centroid
        col = int((lon_c - transform.c) / transform.a)
        row = int((lat_c - transform.f) / transform.e)
        
        col = max(0, min(col, dem.shape[1] - 1))
        row = max(0, min(row, dem.shape[0] - 1))
        
        depth_val = float(dem[row, col]) if np.isfinite(dem[row, col]) else -20.0
        tri_val = float(tri[row, col]) if np.isfinite(tri[row, col]) else 1.0
        bpi_val = float(bpi[row, col]) if np.isfinite(bpi[row, col]) else 0.2
        
        # Calculate custom confidence score based on TRI and BPI
        score = 50.0 + (bpi_val * 20.0) + ((tri_val - 1.0) * 30.0)
        score = int(max(30, min(95, score)))  # clamp
        
        if score >= 70:
            cls = "HIGH CONFIDENCE — Likely real reef"
            notes = f"High rugosity (TRI={tri_val:.2f}) and positive BPI ({bpi_val:.2f}m) — rocky outcrop confirmed"
        elif score >= 50:
            cls = "MODERATE — Needs verification"
            notes = f"Moderate rugosity (TRI={tri_val:.2f}) — possible low-profile reef or hard substrate"
        else:
            cls = "LOW — Probably noise/sand ripple"
            notes = "Weak morphological indices — likely sediment feature"
            
        properties.append({
            "id": idx,
            "confidence_score": score,
            "validation_class": cls,
            "validation_notes": notes,
            "area_m2": area_m2,
            "depth_mean": depth_val,
            "tri_mean": tri_val,
            "bpi_mean": bpi_val
        })
        
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geoms, crs=crs)
    # Convert to EPSG:4326 for Leaflet
    gdf = gdf.to_crs(epsg=4326)
    
    geojson_path = os.path.join(_PROJECT_ROOT, "reef_Output_Master", "reef_output_v3", "regional_mounds.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"✓ Saved {len(gdf)} validated regional mounds to {geojson_path}")
    
    # Return bounds for layers_meta
    return bounds_wgs84

if __name__ == "__main__":
    export_regional_data()
