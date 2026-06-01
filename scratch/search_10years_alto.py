import sys
import os
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
from datetime import datetime

_PROJECT_ROOT = "/Users/ssoares/Downloads/PI-PROJE/reef_imagery_pipeline"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.reef_ml_predictor_acolite import make_snr_map, estimate_kd_bandratio
from src.ranking_model import predict_score

lat, lon = 37.058150, -8.209820
depth = 16.0
buffer_m = 500
years = 10
max_cloud_stac = 30
local_cloud_reject = 15.0

print("=" * 80)
print("  PEDRA DO ALTO — 10-YEAR HISTORICAL SEARCH (Sentinel-2 L2A)")
print(f"  GPS: {lat:.6f} N, {abs(lon):.6f} W | Depth: {depth}m | Buffer: +/-{buffer_m}m")
print("=" * 80)

# Step 1: STAC Search
print("\n[1] Searching Sentinel-2 L2A via Planetary Computer STAC (Last 10 Years)...")
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

end_date = datetime.now()
start_date = datetime(end_date.year - years, 1, 1)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
    query={"eo:cloud_cover": {"lt": max_cloud_stac}}
)

items = list(search.items())
print(f"    Found {len(items)} scenes with <{max_cloud_stac}% STAC cloud cover")

if not items:
    print("    No scenes found. Exiting.")
    sys.exit(1)

# Build dataframe
stac_data = []
for item in items:
    props = item.properties
    if props.get("s2:nodata_pixel_percentage", 100) > 20:
        continue
    stac_data.append({
        "date_str": item.datetime.strftime("%Y-%m-%d"),
        "date": pd.Timestamp(item.datetime.date()),
        "cloud_stac": props.get("eo:cloud_cover", 100),
        "item": item,
    })

df_stac = pd.DataFrame(stac_data)
df_stac = df_stac.sort_values("cloud_stac").drop_duplicates("date_str", keep="first")
print(f"    Unique dates to evaluate: {len(df_stac)}")

# Smart stratified monthly selection to make the 10-year query fast and highly diverse
df_stac["month"] = df_stac["date"].dt.month
per_month = 4 # Take top 4 clear days per month over the 10 years (total ~48 candidates)
candidates = (
    df_stac.sort_values("cloud_stac")
    .groupby("month", group_keys=False)
    .head(per_month)
    .head(50) # Cap at 50 candidate dates for processing efficiency
)
print(f"    Stratified selection: Evaluating top {len(candidates)} historical candidates...")

def check_local_cloud(b02, b08=None):
    cloud_b02 = b02 > 0.18
    if b08 is not None:
        cloud_b08_safe = (b08 > 0.12) & (b02 > 0.12)
    else:
        cloud_b08_safe = cloud_b02
    return float(max(cloud_b02.mean(), cloud_b08_safe.mean())) * 100

# Step 2: Download local windows and compute BVI
print("\n[2] Processing candidate dates via VSI streaming and ML BVI scoring...")
results = []
env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")

for idx, row in candidates.iterrows():
    item = row["item"]
    date_str = row["date_str"]
    sys.stdout.write(f"\r    [{len(results)+1}/{len(candidates)}] Processing {date_str}...      ")
    sys.stdout.flush()
    
    try:
        with env:
            with rasterio.open(item.assets["B02"].href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                window = from_bounds(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m, src.transform)
                b02_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(item.assets["B03"].href) as src:
                b03_arr = src.read(1, window=window).astype(np.float32)
            b08_arr = None
            if "B08" in item.assets:
                with rasterio.open(item.assets["B08"].href) as src:
                    b08_arr = src.read(1, window=window).astype(np.float32)
    except Exception:
        continue
        
    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03 = np.clip(b03_arr / 10000.0, 0, 1.5)
    b08 = np.clip(b08_arr / 10000.0, 0, 1.5) if b08_arr is not None else None
    
    if b02.max() == 0 or np.all(np.isnan(b02)):
        continue
        
    local_cloud_pct = check_local_cloud(b02, b08)
    if local_cloud_pct > local_cloud_reject:
        continue
        
    # Sunglint correction
    p95_b02 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
    p95_b03 = np.percentile(b03[b03 > 0], 95) if np.any(b03 > 0) else 0
    b02_corr = np.clip(b02 - 0.8 * p95_b02 * 0.05, 0, 1.0)
    b03_corr = np.clip(b03 - 0.8 * p95_b03 * 0.05, 0, 1.0)
    
    # Metrics
    snr_map = make_snr_map(b02_corr, window=5)
    snr_mean = float(np.nanmean(snr_map))
    
    macro = cv2.GaussianBlur(b02_corr, (9, 9), 0)
    laplacian = cv2.Laplacian(macro, cv2.CV_32F)
    laplacian_var = float(np.var(laplacian))
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = float(np.mean(grad_mag))
    benthic_contrast = laplacian_var * 1e6 + sobel_mean * 100
    
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
    
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0
        
    kd_est, _ = estimate_kd_bandratio(b02_corr, b03_corr, 0.045)
    raw_mean = float(np.mean(b02[b02 > 0])) if np.any(b02 > 0) else 0
    dyn_range = float(np.max(b02_corr) - np.min(b02_corr)) if np.any(b02_corr) else 0.008
    
    features = {
        "benthic_contrast": benthic_contrast,
        "snr": snr_mean,
        "fft_clean": fft_cleanliness,
        "edge_entropy": edge_entropy,
        "dyn_range": dyn_range,
        "signal": raw_mean,
        "cloud_cover": local_cloud_pct,
    }
    prediction = predict_score(features)
    
    results.append({
        "date": date_str,
        "cloud_stac": row["cloud_stac"],
        "local_cloud_pct": local_cloud_pct,
        "snr": snr_mean,
        "benthic_contrast": benthic_contrast,
        "fft_cleanliness": fft_cleanliness,
        "edge_entropy": edge_entropy,
        "kd_mean": kd_est,
        "raw_mean": raw_mean,
        "dyn_range": dyn_range,
        "BVI": prediction["score"],
        "mode": prediction["mode"]
    })

print(f"\r    Processed {len(results)} scenes successfully.          ")

# Sort and output
df_res = pd.DataFrame(results).sort_values("BVI", ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("  🏆 PEDRA DO ALTO — TOP HISTORICAL DATES (10-YEAR ARCHIVE)")
print("=" * 80)
print(df_res.head(15).to_string(index=False, columns=["date", "BVI", "mode", "local_cloud_pct", "snr", "kd_mean", "fft_cleanliness"]))
print("=" * 80)
