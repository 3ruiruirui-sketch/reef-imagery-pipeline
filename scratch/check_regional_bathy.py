#!/usr/bin/env python3
import os
import sys
import numpy as np
import requests
import rasterio
from rasterio.features import shapes as rasterio_shapes
from scipy.ndimage import uniform_filter, generic_filter, label as ndimage_label

# Set up project root on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from src.bathy_calibrator import fetch_isobaths_for_bbox

def download_emodnet_large(w, s, e, n, out_path, target_m=115):
    """Download large regional EMODnet bathymetry."""
    # Approximate width and height in pixels
    px_w = int(abs(e - w) * 111320.0 / target_m)
    px_h = int(abs(n - s) * 111320.0 / target_m)
    
    url = "https://ows.emodnet-bathymetry.eu/wcs"
    params = {
        "SERVICE":  "WCS",
        "VERSION":  "1.0.0",
        "REQUEST":  "GetCoverage",
        "COVERAGE": "emodnet:mean",
        "CRS":      "EPSG:4326",
        "BBOX":     f"{w:.6f},{s:.6f},{e:.6f},{n:.6f}",
        "WIDTH":    str(px_w),
        "HEIGHT":   str(px_h),
        "FORMAT":   "image/tiff",
    }
    
    print(f"Downloading EMODnet bathymetry ({px_w}x{px_h} px)...")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"✓ Saved bathymetry to {out_path}")
    return out_path

def analyze_benthic_structures():
    # Regional bbox: Carvoeiro (-8.45) to Vilamoura (-8.11)
    w, s, e, n = -8.4500, 37.0200, -8.1100, 37.0900
    
    scratch_dir = os.path.dirname(os.path.abspath(__file__))
    bathy_tif = os.path.join(scratch_dir, "regional_emodnet.tif")
    
    # Download
    download_emodnet_large(w, s, e, n, bathy_tif)
    
    # Read raster
    with rasterio.open(bathy_tif) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
        
    print(f"Loaded bathymetry array: {dem.shape}")
    print(f"Depth range: min={np.nanmin(dem):.1f}m, max={np.nanmax(dem):.1f}m")
    
    # Filter depth: 10m to 40m (represented as negative)
    # EMODnet uses negative below sea level, let's verify
    is_neg = (np.nanmin(dem) < 0)
    if not is_neg:
        print("Bathy is positive, negating for analysis...")
        dem = -dem
        
    depth_mask = (dem >= -40.0) & (dem <= -10.0) & np.isfinite(dem)
    print(f"Pixels in 10-40m depth window: {depth_mask.sum()} / {dem.size}")
    
    # Calculate slope and TRI (rugosity)
    # Average pixel resolution
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    dx_m = res_x * 111320.0 * np.cos(np.radians(37.05))
    dy_m = res_y * 111320.0
    
    # Fill nan with 0 for filters
    dem_filled = np.where(np.isfinite(dem), dem, 0.0)
    
    # Slope
    dz_dx = np.gradient(dem_filled, dx_m, axis=1)
    dz_dy = np.gradient(dem_filled, dy_m, axis=0)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope = np.where(np.isfinite(dem), slope, np.nan)
    
    # TRI (rugosity)
    def _tri_kernel(x):
        centre = x[len(x) // 2]
        diffs = x - centre
        diffs[len(x) // 2] = 0.0
        return float(np.sqrt(np.sum(diffs**2) / 8.0))
        
    tri = generic_filter(dem_filled, _tri_kernel, size=3, mode="nearest")
    tri = np.where(np.isfinite(dem), tri, np.nan)
    
    # BPI broad (15x15) to find elevated mounds relative to surroundings
    mean_broad = uniform_filter(dem_filled, size=15, mode="nearest")
    bpi = dem - mean_broad
    bpi = np.where(np.isfinite(dem), bpi, np.nan)
    
    # Detect structures (high rugosity TRI > 70th percentile of valid cells in 10-40m depth)
    tri_valid = tri[depth_mask & np.isfinite(tri)]
    tri_thresh = np.percentile(tri_valid, 75) if len(tri_valid) > 0 else 1.0
    
    bpi_valid = bpi[depth_mask & np.isfinite(bpi)]
    bpi_thresh = np.percentile(bpi_valid, 75) if len(bpi_valid) > 0 else 0.2
    
    candidates = depth_mask & (tri >= tri_thresh) & (bpi >= bpi_thresh)
    
    # Connected components
    labelled, n_features = ndimage_label(candidates)
    print(f"Detected {n_features} unique benthic mounds/structures in this depth window.")
    
    # Analyze each structure and sort by score/rugosity
    structures = []
    for i in range(1, n_features + 1):
        mask = (labelled == i)
        area_px = mask.sum()
        if area_px < 2:  # Ignore tiny noise
            continue
            
        area_m2 = area_px * dx_m * dy_m
        avg_depth = float(np.mean(dem[mask]))
        avg_tri = float(np.mean(tri[mask]))
        max_tri = float(np.max(tri[mask]))
        avg_bpi = float(np.mean(bpi[mask]))
        
        # Center of mass (pixels)
        rows, cols = np.where(mask)
        crow, ccol = np.mean(rows), np.mean(cols)
        
        # Convert to lat/lon
        lon_c = transform.c + ccol * transform.a + crow * transform.b
        lat_c = transform.f + ccol * transform.d + crow * transform.e
        
        structures.append({
            "id": i,
            "lon": lon_c,
            "lat": lat_c,
            "area_m2": area_m2,
            "depth": avg_depth,
            "tri_mean": avg_tri,
            "tri_max": max_tri,
            "bpi_mean": avg_bpi
        })
        
    # Sort structures by rugosity (TRI)
    structures.sort(key=lambda s: s["tri_mean"], reverse=True)
    
    print("\n" + "="*80)
    print(f"TOP 10 BENTHIC STRUCTURES / REEFS DETECTED (CARVOEIRO TO VILAMOURA, 10-40M)")
    print("="*80)
    for idx, s in enumerate(structures[:10]):
        # Identify approximate coastal spot based on longitude
        # Carvoeiro is around -8.44, Gale is -8.23, Albufeira is -8.21, Vilamoura is -8.12
        lon = s["lon"]
        zone = "Desconhecido"
        if lon < -8.38:
            zone = "Carvoeiro / Benagil"
        elif lon < -8.28:
            zone = "Armação de Pêra / Senhora da Rocha"
        elif lon < -8.215:
            zone = "Galé / Pedra do Alto Reef"
        elif lon < -8.18:
            zone = "Albufeira / Oura Reefs"
        elif lon < -8.13:
            zone = "Olhos de Água"
        else:
            zone = "Vilamoura / Falésia"
            
        print(f"\nStructure #{idx+1} in {zone}:")
        print(f"  Coordinates: Lat {s['lat']:.5f}°, Lon {s['lon']:.5f}°")
        print(f"  Mean Depth : {s['depth']:.1f} meters")
        print(f"  Rugosity   : TRI_mean={s['tri_mean']:.2f} (max={s['tri_max']:.2f})")
        print(f"  BPI Mounding: {s['bpi_mean']:.2f} meters above surrounding seafloor")
        print(f"  Est. Area  : {s['area_m2']:,.0f} m²")

if __name__ == "__main__":
    analyze_benthic_structures()
