#!/usr/bin/env python3
"""
calibrate_and_map.py
========================================================================
Main absolute calibration script for Sentinel-2 SDB.
Takes relative depth proxies (Stumpf ratio) and converts them into:
  1. Absolute metric depth maps (in meters) with plasma colormap & scale bar.
  2. Overlaid depth contours (2m, 5m, 10m, 15m, 20m) on the green reef image.

Data sources:
  - DGRM / IH Isobath REST Service (Primary)
  - EMODnet Bathymetry local TIFF (Automatic Fallback)
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from scratch.generate_enhanced_v2 import process_site, SITES, OUT_DIR, green_reef_cmap, blue_depth_cmap
from src.bathy_calibrator import fetch_isobaths_for_bbox, calibrate_stumpf_from_isobaths, classify_benthic_zone

# Configuração
SITE_KEY   = "pedra_do_alto"  # Foco: Pedra do Alto
DATE_STR   = "2025-09-25"
BUFFER_M   = 600   # 1.2 km × 1.2 km
DPI        = 300

def calibrate_from_emodnet(b02_enh, b03_enh, s2_shape, s2_transform, s2_crs, emodnet_path="scratch/regional_emodnet.tif", n=1000.0):
    """
    Fallback calibration using regional EMODnet bathymetry.
    Reprojects EMODnet TIFF to S2 crop grid and fits Stumpf coefficients.
    """
    print(f"🌍 A ler EMODnet de {emodnet_path}...")
    with rasterio.open(emodnet_path) as src:
        # Read and invert (EMODnet depths are negative below sea level)
        emodnet_data = -src.read(1).astype(np.float32)
        
        # Reproject to S2 grid
        bathy_resampled = np.empty(s2_shape, dtype=np.float32)
        reproject(
            source=emodnet_data,
            destination=bathy_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
            resampling=Resampling.bilinear
        )
        
    # Stumpf ratio calculation
    eps = 1e-6
    X = np.log(n * b02_enh + eps) / np.log(n * b03_enh + eps)
    
    # Filter valid pixels: depth between 2m and 30m, and valid S2 reflectances
    valid_mask = (bathy_resampled >= 2.0) & (bathy_resampled <= 30.0) & (b02_enh > 0.001) & (b03_enh > 0.001)
    
    X_vals = X[valid_mask]
    depth_vals = bathy_resampled[valid_mask]
    
    if len(X_vals) < 50:
        return -16.0, 20.0, {
            "n_samples": len(X_vals),
            "calibrated": False,
            "reason": "insufficient_pixels_in_depth_range",
            "method": "emodnet_fallback"
        }
        
    m1_raw, m0_raw = np.polyfit(X_vals, depth_vals, 1)
    
    # Apply physical constraints
    m0 = float(np.clip(m0_raw, -60.0, 5.0))
    m1 = float(np.clip(m1_raw, 10.0, 70.0))
    
    depth_pred = m1 * X_vals + m0
    rmse = float(np.sqrt(np.mean((depth_pred - depth_vals) ** 2)))
    
    print(f"   [EMODnet Calibration]: m0={m0:.2f}, m1={m1:.2f}, RMSE={rmse:.2f}m ({len(X_vals)} samples)")
    
    return m0, m1, {
        "n_samples": len(X_vals),
        "calibrated": True,
        "method": "emodnet_pixel_regression",
        "m0": round(m0, 4),
        "m1": round(m1, 4),
        "rmse_m": round(rmse, 4)
    }

def main():
    site = SITES[SITE_KEY]
    label = site["label"]
    
    # 1. Process Sentinel-2 crop
    print("📡 A descarregar e processar Sentinel-2...")
    res = process_site(SITE_KEY, DATE_STR, BUFFER_M, OUT_DIR, DPI)
    
    b02_enh = res["b02_enh"]
    b03_enh = res["b03_enh"]
    b02_final = res["b02_final"]
    ratio = res["ratio"]
    bounds_wgs84 = res["bounds_wgs84"]
    s2_transform = res["s2_transform"]
    s2_crs = res["s2_crs"]
    
    min_lat, min_lon, max_lat, max_lon = bounds_wgs84
    
    # 2. Fetch ground truth isobaths (IH Portal) with EMODnet fallback
    features = []
    use_emodnet = False
    
    print("\n📡 A obter isóbatas IH (DGRM API)...")
    try:
        features = fetch_isobaths_for_bbox(min_lon, min_lat, max_lon, max_lat)
        if not features:
            print("   ⚠️  Nenhuma isóbata IH disponível no AOI. A usar EMODnet...")
            use_emodnet = True
    except Exception as e:
        print(f"   ⚠️  API IH/DGRM offline ou falhou ({e}). A usar EMODnet...")
        use_emodnet = True
        
    # 3. Perform Calibration
    m0, m1, cal_report = -16.0, 20.0, {}
    
    if use_emodnet:
        emodnet_path = os.path.join(_PROJECT_ROOT, "scratch", "regional_emodnet.tif")
        # Ensure fallback file is downloaded
        if not os.path.exists(emodnet_path):
            from scratch.check_regional_bathy import download_emodnet_large
            download_emodnet_large(-8.4500, 37.0200, -8.1100, 37.0900, emodnet_path)
        
        m0, m1, cal_report = calibrate_from_emodnet(
            b02_enh, b03_enh, b02_final.shape, s2_transform, s2_crs, emodnet_path
        )
    else:
        # IH OLS Regression
        print("🌍 A calibrar com isóbatas IH...")
        m0, m1, cal_report = calibrate_stumpf_from_isobaths(
            b02_enh, b03_enh, features, bounds_wgs84
        )
        
    # 4. Apply absolute depth model
    # depth = m1 * ratio + m0
    depth_m = m1 * ratio + m0
    
    # Classify benthic zone of the center spot
    lat, lon = site["lat"], site["lon"]
    zone_report = {}
    if features:
        zone_report = classify_benthic_zone(lon, lat, features)
        
    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    slug = SITE_KEY.replace("_", "-")
    date_slug = DATE_STR.replace("-", "")
    
    print("\n🎨 A renderizar mapas métricos absolutos...")
    
    # Mask invalid/deep areas
    # 0 to 30m is the transparent optical window
    masked_depth = np.ma.masked_where((depth_m < 0.0) | (depth_m > 30.0) | np.isnan(depth_m), depth_m)
    
    # A. Map 1: Absolute Bathymetry Grid with Colorbar
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="#001108")
    im = ax.imshow(masked_depth, cmap="plasma_r", vmin=0, vmax=25, interpolation="bilinear")
    ax.set_title(f"{label} · Batimetria Absoluta Metrica (IH) · {DATE_STR}", color="white", fontsize=11, pad=10, fontfamily="monospace")
    ax.axis("off")
    
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
    cbar.set_label("Profundidade (metros reais)", color="white", fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")
    
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    out_bathy = OUT_DIR / f"{slug}_{date_slug}_bathy_absolute.png"
    fig.savefig(out_bathy, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Guardado: {out_bathy.name} ({os.path.getsize(out_bathy)/1e6:.2f} MB)")
    
    # B. Map 2: Depth contours (isobaths) overlaid on Green Reef view
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#001108")
    ax.imshow(b02_final, cmap=green_reef_cmap, interpolation="bilinear", vmin=0, vmax=1)
    
    levels = [2.0, 5.0, 10.0, 15.0, 20.0]
    valid_levels = [l for l in levels if np.nanmin(depth_m) < l < np.nanmax(depth_m)]
    
    if valid_levels:
        contours = ax.contour(depth_m, levels=valid_levels, colors="#ff3366", linewidths=1.2, alpha=0.85)
        ax.clabel(contours, inline=True, fontsize=8, fmt="%1.0fm", colors="white")
        
    ax.set_title(f"{label} · Contornos de Profundidade · {DATE_STR}", color="white", fontsize=10, pad=6, fontfamily="monospace")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.96)
    
    out_contours = OUT_DIR / f"{slug}_{date_slug}_bathy_contours.png"
    fig.savefig(out_contours, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  ✅ Guardado: {out_contours.name} ({os.path.getsize(out_contours)/1e6:.2f} MB)")
    
    # C. Save Calibration Report JSON
    report_data = {
        "site": SITE_KEY,
        "date": DATE_STR,
        "m0": m0,
        "m1": m1,
        "isobaths_available": [f["depth"] for f in features] if features else "EMODnet Fallback Grid",
        "calibration": cal_report,
        "benthic_zone": zone_report
    }
    
    out_report = OUT_DIR / f"{slug}_{date_slug}_calibration_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Guardado Relatório: {out_report.name}")
    
    print("\n🎉 Calibração Concluída com Sucesso!")

if __name__ == "__main__":
    main()
