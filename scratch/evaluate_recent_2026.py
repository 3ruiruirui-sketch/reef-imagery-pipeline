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

def check_local_cloud(b02, b08=None):
    cloud_b02 = b02 > 0.18
    if b08 is not None:
        cloud_b08_safe = (b08 > 0.12) & (b02 > 0.12)
    else:
        cloud_b08_safe = cloud_b02
    return float(max(cloud_b02.mean(), cloud_b08_safe.mean())) * 100

def evaluate_date_site(lat, lon, date_str, site_name, buffer_m=500):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    target_date = pd.to_datetime(date_str).date()
    next_day = target_date + pd.Timedelta(days=1)
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{target_date.isoformat()}/{next_day.isoformat()}"
    )
    items = list(search.items())
    if not items:
        return None
    
    item = items[0]
    b02_href = item.assets["B02"].href
    b03_href = item.assets["B03"].href
    b08_href = item.assets["B08"].href if "B08" in item.assets else None
    
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    try:
        with env:
            with rasterio.open(b02_href) as src:
                tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = tf.transform(lon, lat)
                window = from_bounds(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m, src.transform)
                b02_arr = src.read(1, window=window).astype(np.float32)
            with rasterio.open(b03_href) as src:
                b03_arr = src.read(1, window=window).astype(np.float32)
            b08_arr = None
            if b08_href:
                with rasterio.open(b08_href) as src:
                    b08_arr = src.read(1, window=window).astype(np.float32)
    except Exception:
        return None
        
    b02 = np.clip(b02_arr / 10000.0, 0, 1.5)
    b03 = np.clip(b03_arr / 10000.0, 0, 1.5)
    b08 = np.clip(b08_arr / 10000.0, 0, 1.5) if b08_arr is not None else None
    
    local_cloud_pct = check_local_cloud(b02, b08)
    if local_cloud_pct > 15:
        return {
            "date": date_str,
            "local_cloud_pct": local_cloud_pct,
            "snr": 0,
            "benthic_contrast": 0,
            "fft_cleanliness": 0,
            "edge_entropy": 0,
            "kd_mean": 0,
            "raw_mean": 0,
            "ml_score": 0.0,
            "status": "Cloudy"
        }
    
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
    
    return {
        "date": date_str,
        "local_cloud_pct": local_cloud_pct,
        "snr": snr_mean,
        "benthic_contrast": benthic_contrast,
        "fft_cleanliness": fft_cleanliness,
        "edge_entropy": edge_entropy,
        "kd_mean": kd_est,
        "raw_mean": raw_mean,
        "ml_score": prediction["score"],
        "status": "OK"
    }

dates_2026 = [
    "2026-05-23", "2026-05-20", "2026-04-18", "2026-04-10", "2026-04-03",
    "2026-03-31", "2026-03-29", "2026-03-24", "2026-03-11", "2026-02-22",
    "2026-02-12", "2026-02-02", "2026-01-18", "2026-01-08"
]

print("=== 2026 SCENES EVALUATION FOR SANTA EULALIA ===")
eulalia_res = []
for d in dates_2026:
    res = evaluate_date_site(37.068978, -8.210328, d, "Pedra de Santa Eulalia")
    if res:
        eulalia_res.append(res)
df_eulalia = pd.DataFrame(eulalia_res).sort_values("ml_score", ascending=False)
print(df_eulalia[["date", "local_cloud_pct", "snr", "kd_mean", "ml_score", "status"]].to_string(index=False))

print("\n=== 2026 SCENES EVALUATION FOR PEDRA DO ALTO ===")
alto_res = []
for d in dates_2026:
    res = evaluate_date_site(37.058150, -8.209820, d, "Pedra do Alto")
    if res:
        alto_res.append(res)
df_alto = pd.DataFrame(alto_res).sort_values("ml_score", ascending=False)
print(df_alto[["date", "local_cloud_pct", "snr", "kd_mean", "ml_score", "status"]].to_string(index=False))
