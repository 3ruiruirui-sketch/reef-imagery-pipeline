import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client
from pathlib import Path
from datetime import datetime as dt
import pandas as pd

lat, lon = 37.068978, -8.210328
date_str = "2025-09-25"
buffer_m = 1000 # 1km buffer to show the coast and ocean naturally!

print("Finding scene for Pedra de Santa Eulalia on 2025-09-25...")
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

def download_band(asset_name):
    print(f"Downloading {asset_name}...")
    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    with env:
        with rasterio.open(item.assets[asset_name].href) as src:
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
            arr = src.read(1, window=window).astype(np.float32)
    return arr

# Download B04 (Red), B03 (Green), B02 (Blue)
r_arr = download_band("B04")
g_arr = download_band("B03")
b_arr = download_band("B02")

# L2A DN -> Reflectance (0-1)
r = np.clip(r_arr / 10000.0, 0, 1.0)
g = np.clip(g_arr / 10000.0, 0, 1.0)
b = np.clip(b_arr / 10000.0, 0, 1.0)

# Stack to RGB
rgb = np.stack([r, g, b], axis=2)

# Normalize to uint8 (0-255)
p2, p98 = np.percentile(rgb, [2, 98])
rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-12) * 255, 0, 255).astype(np.uint8)

# OpenCV expects BGR
bgr = cv2.cvtColor(rgb_stretched, cv2.COLOR_RGB2BGR)

# Apply contrast enhancement (CLAHE on L channel of LAB) to make the underwater geology pop out naturally!
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
l, a, b_ch = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b_ch))
enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Resize to 2000px and save to artifacts
h, w, c = enhanced.shape
target_width = 2000
scale = target_width / w
new_h = int(h * scale)
resized = cv2.resize(enhanced, (target_width, new_h), interpolation=cv2.INTER_AREA)

dst_path = "/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965/pedra_santa_eulalia_natural_s2.png"
cv2.imwrite(dst_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(f"✅ Success! Saved natural true color Sentinel-2 image: {dst_path} (size: {os.path.getsize(dst_path)/1e6:.2f} MB)")
