import cv2
import os

src_path = "/Users/ssoares/Downloads/PI-PROJE/outputs/dive_sites_1200dpi/pedra_santa_eulalia_pleiades_30cm_blue_1200dpi.png"
dst_path = "/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965/pedra_santa_eulalia_pleiades_30cm_compressed.png"

print(f"Reading source image: {src_path} (size: {os.path.getsize(src_path)/1e6:.1f} MB)...")
img = cv2.imread(src_path)

if img is None:
    print("❌ Failed to read source image!")
else:
    h, w, c = img.shape
    print(f"Loaded image shape: {w}x{h} ({c} channels)")
    
    # Let's resize it slightly to 2000px width/height for fast loading, while keeping the exceptional high detail
    target_width = 2000
    scale = target_width / w
    new_h = int(h * scale)
    print(f"Resizing to: {target_width}x{new_h}...")
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    
    # Save with high compression
    print(f"Saving compressed image to: {dst_path}...")
    cv2.imwrite(dst_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Success! Compressed image size: {os.path.getsize(dst_path)/1e6:.2f} MB")
