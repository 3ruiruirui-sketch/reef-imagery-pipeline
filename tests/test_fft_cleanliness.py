import numpy as np
import cv2
import sys
from src.enhancer import fetch_vsi_patch

lat, lon = 37.05811, -8.20978

def analyze_fft(date_str):
    try:
        # fetch 1000m buffer (100x100 pixels at 10m res)
        b02_ref = fetch_vsi_patch(lat, lon, date_str, buffer_m=500.0)
    except Exception as e:
        return {"error": str(e)}

    # Remove 0s (nodata)
    valid = b02_ref[b02_ref > 0]
    if len(valid) == 0:
        return {"error": "Empty data"}

    # Basic stats
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    
    # 2D FFT
    f_transform = np.fft.fft2(b02_ref)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    
    h, w = b02_ref.shape
    cy, cx = h // 2, w // 2
    
    # Define masks for low frequencies (macro structures) and high frequencies (noise/waves)
    # Since resolution is 10m, a 100x100 patch is 1000m x 1000m.
    # Radius of 5 pixels corresponds to wavelengths > 200m (macro geological structures)
    r_low = 5
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx)**2 + (y - cy)**2 <= r_low**2
    
    # Radius of 15+ pixels corresponds to wavelengths < 66m (surface waves, ripples, glint)
    r_high = 15
    mask_high = (x - cx)**2 + (y - cy)**2 >= r_high**2
    
    low_power = np.sum(power[mask_low])
    high_power = np.sum(power[mask_high])
    
    # Cleanliness Ratio
    cleanliness_ratio = low_power / (high_power + 1e-12)
    
    # Contrast Benthico (Laplacian of Gaussian on a gently blurred image to focus on macro structures)
    b02_blur = cv2.GaussianBlur(b02_ref, (5, 5), 0)
    lap = cv2.Laplacian(b02_blur, cv2.CV_32F)
    lap_var = float(np.var(lap)) * 1000000
    
    return {
        "mean": mean_val,
        "std": std_val,
        "low_power": low_power,
        "high_power": high_power,
        "cleanliness_ratio": cleanliness_ratio,
        "lap_var": lap_var
    }

dates = ["2025-09-25", "2025-09-02", "2023-09-01", "2024-09-30", "2022-09-26", "2019-09-07", "2020-09-06", "2017-09-02", "2020-09-26"]
print(f"{'Date':12} | {'Mean':7} | {'Std':7} | {'LapVar':8} | {'LowPower':9} | {'HighPower':9} | {'CleanRatio':10}")
print("-" * 80)
for d in dates:
    res = analyze_fft(d)
    if "error" in res:
        print(f"{d:12} | Error: {res['error']}")
    else:
        print(f"{d:12} | {res['mean']:.5f} | {res['std']:.5f} | {res['lap_var']:8.2f} | {res['low_power']:9.1e} | {res['high_power']:9.1e} | {res['cleanliness_ratio']:10.2f}")
