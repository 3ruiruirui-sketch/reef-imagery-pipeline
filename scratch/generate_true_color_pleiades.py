import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import cv2
import numpy as np
import os

def crop_and_enhance_wms(tif_path, lat, lon, site_name, out_name, buffer_m=150):
    print(f"\nProcessing {site_name} from WMS: {tif_path}...")
    if not os.path.exists(tif_path):
        print(f"❌ File does not exist: {tif_path}")
        return
        
    with rasterio.open(tif_path) as src:
        # Transform GPS coordinates to WMS projected CRS
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        
        # Define window (e.g. 300m x 300m)
        window = from_bounds(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m, src.transform)
        
        # Read RGB bands
        # WMS True Color is usually 3 bands (R, G, B) or 4 bands (R, G, B, A)
        bands = []
        for i in range(1, min(src.count + 1, 4)):
            bands.append(src.read(i, window=window))
            
    if not bands:
        print("❌ No bands read!")
        return
        
    # Stack bands to RGB
    rgb = np.stack(bands, axis=2)
    print(f"Original crop shape: {rgb.shape}")
    
    # Normalize if necessary (usually uint8 0-255)
    if rgb.dtype != np.uint8:
        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
        
    # Standard WMS is in RGB, OpenCV expects BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Apply standard contrast enhancement (CLAHE on L channel of LAB color space)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Save the natural true color enhanced crop
    dst_dir = "/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965"
    dst_path = os.path.join(dst_dir, out_name)
    cv2.imwrite(dst_path, enhanced_bgr)
    print(f"Saved enhanced crop: {dst_path} (size: {os.path.getsize(dst_path)/1e6:.2f} MB)")
    
# True color WMS path
wms_true_color = "/Users/ssoares/Downloads/PI-PROJE/reef_imagery_pipeline/outputs/pleiades_neo/ortosat2023_true_color.tif"

# Pedra de Santa Eulália
crop_and_enhance_wms(wms_true_color, 37.068978, -8.210328, "Pedra de Santa Eulalia", "pedra_santa_eulalia_true_color_natural.png")

# Pedra do Alto (Corrected)
crop_and_enhance_wms(wms_true_color, 37.058150, -8.209820, "Pedra do Alto", "pedra_do_alto_true_color_natural.png")
