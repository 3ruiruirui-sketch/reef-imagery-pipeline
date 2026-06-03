import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
from skimage.restoration import denoise_nl_means, estimate_sigma
from pathlib import Path
from datetime import datetime as dt
import pandas as pd

green_reef_cmap = LinearSegmentedColormap.from_list(
    "green_reef", ["#001100", "#002d08", "#004d1b", "#007a33", "#22b15b", "#6ce08c", "#baffd0"]
)

lat, lon = 37.058150, -8.209820
date_str = "2025-03-29"
buffer_m = 500

print(f"Finding scene for Pedra do Alto on {date_str}...")
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
target_dt = dt.strptime(date_str, "%Y-%m-%d").date()
next_day = target_dt + pd.Timedelta(days=1)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime=f"{target_dt.isoformat()}/{next_day.isoformat()}"
)
items = list(search.items())
if not items:
    print("❌ No Sentinel-2 scene found!")
    sys.exit(1)

item = items[0]
print(f"Scene found: {item.id}")

print("Downloading B02 (Blue band) window...")
env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
with env:
    with rasterio.open(item.assets["B02"].href) as src:
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        
        min_x, min_y = x - buffer_m, y - buffer_m
        max_x, max_y = x + buffer_m, y + buffer_m
        
        inv_transform = ~src.transform
        col_min, row_max = inv_transform * (min_x, min_y)
        col_max, row_min = inv_transform * (max_x, max_y)
        
        window = Window(
            col_off=int(col_min),
            row_off=int(row_min),
            width=int(col_max - col_min),
            height=int(row_max - row_min)
        )
        b02_arr = src.read(1, window=window).astype(np.float32)

b02 = np.clip(b02_arr / 10000.0, 0, 1.5)

# 1. Stretches RAW DIRECTLY
print("Generating raw version...")
p2, p98 = np.percentile(b02[b02 > 0], [2, 98]) if np.any(b02 > 0) else (0, 1)
img_stretch_raw = np.clip((b02 - p2) / (p98 - p2 + 1e-12), 0, 1)

fig, ax = plt.subplots(figsize=(6, 6), facecolor="#001100")
ax.imshow(img_stretch_raw, cmap=green_reef_cmap, interpolation="bilinear")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

temp_raw_path = "scratch_alto_best10y_raw.png"
fig.savefig(temp_raw_path, dpi=1200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
plt.close(fig)

# 2. Enhanced version
print("Generating enhanced version...")
p95 = np.percentile(b02[b02 > 0], 95) if np.any(b02 > 0) else 0
b02_corr = np.clip(b02 - 0.8 * p95 * 0.05, 0, 1.0)

sigma_est = np.mean(estimate_sigma(b02_corr))
b02_denoised = denoise_nl_means(b02_corr, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)

b02_16 = np.clip(b02_denoised * 65535, 0, 65535).astype(np.uint16)
clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(4, 4))
b02_clahe = clahe.apply(b02_16).astype(np.float32) / 65535.0

b02_enhanced = b02_denoised * 0.5 + b02_clahe * 0.5

p2, p98 = np.percentile(b02_enhanced[b02_enhanced > 0], [2, 98]) if np.any(b02_enhanced > 0) else (0, 1)
img_stretch_enh = np.clip((b02_enhanced - p2) / (p98 - p2 + 1e-12), 0, 1)

fig, ax = plt.subplots(figsize=(6, 6), facecolor="#001100")
ax.imshow(img_stretch_enh, cmap=green_reef_cmap, interpolation="bilinear")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

temp_enh_path = "scratch_alto_best10y_enh.png"
fig.savefig(temp_enh_path, dpi=1200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
plt.close(fig)

# Compress both
print("Compressing and saving to artifacts...")
for temp_path, out_name in [(temp_raw_path, "pedra_do_alto_best10y_green_20250329_raw.png"), (temp_enh_path, "pedra_do_alto_best10y_green_20250329_enhanced.png")]:
    img = cv2.imread(temp_path)
    h, w, c = img.shape
    target_width = 2000
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    
    dst_path = f"/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965/{out_name}"
    cv2.imwrite(dst_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"✅ Saved: {dst_path} ({os.path.getsize(dst_path)/1e6:.2f} MB)")
