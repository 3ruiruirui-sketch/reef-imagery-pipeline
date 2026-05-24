import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from datetime import datetime
import warnings
from src.reef_ml_predictor_acolite import make_snr_map

warnings.filterwarnings("ignore")

def fetch_vsi_patch(lat, lon, date_str, buffer_m=500.0):
    catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        intersects={'type': 'Point', 'coordinates': [lon, lat]},
        datetime=f"{date_str}/{date_str}"
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No STAC item found for {date_str}")
        
    item = items[0]
    b02_href = item.assets["B02"].href
    
    env = rasterio.Env(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR')
    with env:
        with rasterio.open(b02_href) as src:
            tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = tf.transform(lon, lat)
            window = from_bounds(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m, src.transform)
            
            b02_arr = src.read(1, window=window).astype(np.float32)
            
    # L2A DN -> Reflectance
    return np.clip(b02_arr / 10000.0, 0, 1.5)

def run_enhancement_pipeline(lat, lon, image_date, target_snr):
    # 1. Fetch raw data
    b02_ref = fetch_vsi_patch(lat, lon, image_date, buffer_m=1000.0)
    
    # 2. Sunglint removal (Empirical for this fast script)
    p95 = np.percentile(b02_ref[b02_ref > 0], 95)
    b02_glint_free = np.clip(b02_ref - 0.8 * p95 * 0.05, 0, 1.0)
    
    # 3. Spatial Denoising (Non-Local Means)
    sigma_est = np.mean(estimate_sigma(b02_glint_free))
    b02_denoised = denoise_nl_means(b02_glint_free, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    
    # 4. Local Contrast Equalization (CLAHE) - REFINED
    # Convert to 16-bit uint for OpenCV CLAHE
    b02_16 = np.clip(b02_denoised * 65535, 0, 65535).astype(np.uint16)
    
    # Use a very gentle clipLimit to prevent SNR destruction
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(4,4))
    b02_clahe = clahe.apply(b02_16)
    
    # Convert back to reflectance
    b02_clahe_float = b02_clahe.astype(np.float32) / 65535.0
    
    # Blend 50/50 with denoised to preserve radiometric balance and SNR
    b02_final = (b02_denoised * 0.5) + (b02_clahe_float * 0.5)
    
    # 5. Evaluate SNR
    snr_map_raw = make_snr_map(b02_ref, window=5)
    snr_map_final = make_snr_map(b02_final, window=5)
    
    snr_mean = float(np.nanmean(snr_map_final))
    snr_median = float(np.nanmedian(snr_map_final))
    percent_useful = float((np.count_nonzero(snr_map_final > 50) / snr_map_final.size) * 100)
    
    warnings_list = []
    if snr_mean > target_snr * 1.5:
        warnings_list.append("CLAHE artifacting detected: Apparent SNR is artificially high, violating radiometric physics.")
        
    return {
        "patch_bounds": "1000x1000m centered",
        "algorithms": ["Empirical Sunglint", "NLM Spatial Denoising", "CLAHE"],
        "snr_mean": snr_mean,
        "snr_median": snr_median,
        "percent_useful": percent_useful,
        "warnings": warnings_list
    }
