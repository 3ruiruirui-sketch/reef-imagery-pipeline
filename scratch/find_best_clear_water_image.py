#!/usr/bin/env python3
import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from datetime import datetime

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client

# Add project root to sys.path to enable imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio

def search_s2_scenes(lat, lon, years=8, max_cloud=15):
    """Search Sentinel-2 L2A STAC catalog for overlap scenes."""
    print(f"[1] Searching Sentinel-2 scenes for last {years} years...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    end_date = datetime.now()
    start_date = datetime(end_date.year - years, 1, 1)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": max_cloud}}
    )

    items = list(search.items())
    print(f"    Found {len(items)} scenes with <{max_cloud}% general cloud cover.")
    
    stac_data = []
    for item in items:
        props = item.properties
        # Filter scenes with excessive nodata pixels
        if props.get("s2:nodata_pixel_percentage", 100) > 20:
            continue
        stac_data.append({
            "date_str": item.datetime.strftime("%Y-%m-%d"),
            "date": pd.Timestamp(item.datetime.date()),
            "cloud_cover": props.get("eo:cloud_cover", 100),
            "sun_elevation": props.get("view:sun_elevation", 45),
            "item": item,
        })
        
    df = pd.DataFrame(stac_data)
    if not df.empty:
        df = df.drop_duplicates("date_str").sort_values("date", ascending=False)
    return df

def analyze_scenes(df_stac, lat, lon, buffer_m=250):
    """Download window buffers and evaluate local clouds and water clarity."""
    print(f"\n[2] Downloading {buffer_m*2}m × {buffer_m*2}m windows via VSI and calculating metrics...")
    
    results = []
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    
    # Evaluate top 40 dates sorted by general cloud cover and sun elevation (higher sun = better water penetration)
    candidates = df_stac.sort_values(by=["cloud_cover", "sun_elevation"], ascending=[True, False]).head(40)
    
    for idx, row in candidates.iterrows():
        item = row["item"]
        date_str = row["date_str"]
        sys.stdout.write(f"\r    Processing {date_str}... ({len(results)} analyzed)         ")
        sys.stdout.flush()

        b02_href = item.assets["B02"].href
        b03_href = item.assets["B03"].href
        b08_href = item.assets["B08"].href if "B08" in item.assets else None

        with env:
            try:
                with rasterio.open(b02_href) as src:
                    tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = tf.transform(lon, lat)
                    window = from_bounds(x - buffer_m, y - buffer_m,
                                         x + buffer_m, y + buffer_m, src.transform)
                    b02_arr = src.read(1, window=window).astype(np.float32)
                with rasterio.open(b03_href) as src:
                    b03_arr = src.read(1, window=window).astype(np.float32)
                b08_arr = None
                if b08_href:
                    with rasterio.open(b08_href) as src:
                        b08_arr = src.read(1, window=window).astype(np.float32)
            except Exception:
                continue

        # Convert DN to reflectance [0.0, 1.0]
        b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
        b03 = np.clip(b03_arr / 10000.0, 0, 1.5)
        
        if b02.max() == 0 or np.all(np.isnan(b02)):
            continue

        # --- Local Cloud Check ---
        # Clouds: B02 > 0.12 (bright) or B08 > 0.12 (NIR high)
        cloud_mask_b02 = b02 > 0.12
        cloud_pct_b02 = float(cloud_mask_b02.mean())
        
        if b08_arr is not None:
            b08 = np.clip(b08_arr / 10000.0, 0, 1.5)
            cloud_pct_b08 = float(((b08 > 0.12) & (b02 > 0.10)).mean())
            local_cloud_pct = max(cloud_pct_b02, cloud_pct_b08)
        else:
            b08 = None
            local_cloud_pct = cloud_pct_b02

        # Convert to percentage
        local_cloud_pct *= 100.0

        # --- Sunglint Correction ---
        p95_b02 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
        p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
        b02_corr = np.clip(b02 - 0.8 * p95_b02 * 0.05, 0, 1.0)
        b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)

        # --- Metrics ---
        # 1. Local SNR
        snr_map = make_snr_map(b02_corr, window=5)
        snr_mean = float(np.nanmean(snr_map))

        # 2. Local Kd (band ratio)
        kd_prior = 0.045
        kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, kd_prior)

        # 3. FFT Calmness
        f_transform = np.fft.fft2(b02_corr)
        f_shift = np.fft.fftshift(f_transform)
        power = np.abs(f_shift) ** 2
        h, w = b02_corr.shape
        cy, cx = h // 2, w // 2
        
        yy, xx = np.ogrid[:h, :w]
        mask_low = (xx - cx)**2 + (yy - cy)**2 <= 5**2
        mask_high = (xx - cx)**2 + (yy - cy)**2 >= 15**2
        low_power = np.sum(power[mask_low])
        high_power = np.sum(power[mask_high])
        fft_cleanliness = float(low_power / (high_power + 1e-12))

        # 4. Edge entropy (complexity)
        macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
        sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mean = float(np.mean(grad_mag))
        laplacian = cv2.Laplacian(macro, cv2.CV_32F)
        laplacian_var = float(np.var(laplacian))
        benthic_contrast = laplacian_var * 1e6 + sobel_mean * 100

        if grad_mag.max() > grad_mag.min():
            grad_norm = ((grad_mag - grad_mag.min()) /
                         (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
            hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
            p = hist / np.sum(hist)
            p = p[p > 0]
            edge_entropy = float(-np.sum(p * np.log2(p)))
        else:
            edge_entropy = 0.0

        # Signal bounds check
        raw_mean = float(np.mean(b02[b02 > 0])) if np.any(b02 > 0) else 0

        results.append({
            "date": date_str,
            "general_cloud": row["cloud_cover"],
            "local_cloud": local_cloud_pct,
            "snr": snr_mean,
            "kd_mean": kd_est,
            "fft_cleanliness": fft_cleanliness,
            "edge_entropy": edge_entropy,
            "benthic_contrast": benthic_contrast,
            "raw_mean": raw_mean,
        })
        
    print(f"\r    Successfully evaluated {len(results)} scenes.       ")
    return pd.DataFrame(results)

def rank_and_report(df, lat, lon):
    """Rank scenes based on local cloudiness and water clarity metrics, and report."""
    if df.empty:
        print("❌ No scenes could be evaluated successfully.")
        return
        
    # Strictly filter for no local clouds (local_cloud <= 1.0%)
    df_passed = df[df["local_cloud"] <= 1.0].copy()
    
    if df_passed.empty:
        print("⚠️ No scenes passed the strict local cloud filter (<=1.0%). Retrying with <=5.0%...")
        df_passed = df[df["local_cloud"] <= 5.0].copy()
        
    if df_passed.empty:
        print("❌ No scenes have low cloud cover at this specific location.")
        return

    # Normalize metrics to compute BVI
    def norm(s):
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-12:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    df_passed["n_clean"] = norm(np.log10(df_passed["fft_cleanliness"].clip(lower=1)))
    df_passed["n_kd"] = norm(1.0 / df_passed["kd_mean"])
    df_passed["n_contrast"] = norm(df_passed["benthic_contrast"])
    df_passed["n_entropy"] = norm(df_passed["edge_entropy"])
    df_passed["n_snr"] = norm(df_passed["snr"])

    # Signal ok filter
    signal_ok = ((df_passed["raw_mean"] >= 0.04) &
                 (df_passed["raw_mean"] <= 0.15)).astype(float)
    signal_ok = signal_ok.replace(0, 0.2)

    df_passed["BVI"] = (
        0.25 * df_passed["n_clean"] +
        0.25 * df_passed["n_kd"] +
        0.20 * df_passed["n_contrast"] +
        0.15 * df_passed["n_entropy"] +
        0.15 * df_passed["n_snr"]
    ) * signal_ok

    df_ranked = df_passed.sort_values("BVI", ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print(f" 🏆 TOP VISIBILITY DAYS FOR COORDINATE: {lat:.6f}°N, {abs(lon):.6f}°W")
    print("="*80)
    print(f"  {'#':>2}  {'Date':<12} {'BVI':>5} {'Local☁':>8} {'Gen☁':>8} {'SNR':>6} {'Kd':>7} {'Clean':>8} {'Signal':>7}")
    print("  " + "-"*76)

    for i, r in df_ranked.head(10).iterrows():
        star = "⭐⭐⭐" if r["BVI"] >= 0.7 else ("⭐⭐" if r["BVI"] >= 0.5 else ("⭐" if r["BVI"] >= 0.3 else ""))
        print(f"  {i+1:>2}. {r['date']:<12} {r['BVI']:.3f} {r['local_cloud']:>6.2f}% {r['general_cloud']:>6.2f}% {r['snr']:>5.1f} {r['kd_mean']:>6.4f} {r['fft_cleanliness']:>7.0f} {r['raw_mean']:>6.3f} {star}")

    best = df_ranked.iloc[0]
    print("\n" + "="*80)
    print(f"  🌟 RECOMENDAÇÃO ABSOLUTA: {best['date']}")
    print(f"     BVI (Índice de Visibilidade): {best['BVI']:.3f}")
    print(f"     Nuvens Locais:              {best['local_cloud']:.2f}% (Sem nuvens sobre este recife!)")
    print(f"     Clareza da Água (Kd):       {best['kd_mean']:.4f} (Excelente transparência!)")
    print(f"     Calmaria da Superfície:     {best['fft_cleanliness']:.0f} (Mínimo ruído de ondas)")
    print(f"     Qualidade do Sinal (SNR):   {best['snr']:.1f}")
    print("="*80)
    
    # Save output CSV
    df_ranked.to_csv("best_clear_water_images.csv", index=False)
    print(f"\n✓ Resultados guardados com sucesso em 'best_clear_water_images.csv'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Find absolute best Sentinel-2 image for a coordinate.")
    p.add_argument("--lat", type=float, default=37.040997)
    p.add_argument("--lon", type=float, default=-8.167219)
    p.add_argument("--buffer", type=int, default=250, help="Window buffer radius in meters")
    p.add_argument("--years", type=int, default=8, help="How many years back to search")
    args = p.parse_args()

    df_scenes = search_s2_scenes(args.lat, args.lon, args.years)
    if not df_scenes.empty:
        df_results = analyze_scenes(df_scenes, args.lat, args.lon, args.buffer)
        rank_and_report(df_results, args.lat, args.lon)
