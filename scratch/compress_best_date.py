import cv2
import os

src_path = "/Users/ssoares/Downloads/PI-PROJE/outputs/dive_sites_1200dpi/pedra_santa_eulalia_2025-09-25_blue_1200dpi.png"
dst_path = "/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965/pedra_santa_eulalia_best_20250925_compressed.png"

print(f"Reading source image: {src_path}...")
img = cv2.imread(src_path)
if img is None:
    print("❌ Failed to read image!")
else:
    h, w, c = img.shape
    print(f"Loaded image size: {w}x{h}")
    target_width = 2000
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Successfully saved to: {dst_path} (size: {os.path.getsize(dst_path)/1e6:.2f} MB)")
