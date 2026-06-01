import cv2
import os

true_color_path = "/Users/ssoares/Downloads/PI-PROJE/reef_imagery_pipeline/outputs/pleiades_neo/ortosat2023_true_color.tif"
false_color_path = "/Users/ssoares/Downloads/PI-PROJE/reef_imagery_pipeline/outputs/pleiades_neo/ortosat2023_false_color.tif"
dst_dir = "/Users/ssoares/.gemini/antigravity-ide/brain/d6246231-ba06-4562-9142-e672e792f965"

def enhance_and_save(src_path, out_name, out_compressed_name):
    if not os.path.exists(src_path):
        print(f"❌ File does not exist: {src_path}")
        return
        
    print(f"Reading image directly using OpenCV: {src_path}...")
    img = cv2.imread(src_path)
    if img is None:
        print("❌ OpenCV failed to read image!")
        return
        
    h, w, c = img.shape
    print(f"Loaded image size: {w}x{h} ({c} channels)")
    
    # Apply standard contrast enhancement (CLAHE on L channel of LAB color space)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Save the full-size true color enhanced crop
    dst_path = os.path.join(dst_dir, out_name)
    cv2.imwrite(dst_path, enhanced)
    print(f"Saved enhanced crop: {dst_path} (size: {os.path.getsize(dst_path)/1e6:.2f} MB)")
    
    # Save a compressed 2000px version for fast rendering in chat
    target_width = 2000
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(enhanced, (target_width, new_h), interpolation=cv2.INTER_AREA)
    
    dst_comp_path = os.path.join(dst_dir, out_compressed_name)
    cv2.imwrite(dst_comp_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Saved compressed crop: {dst_comp_path} (size: {os.path.getsize(dst_comp_path)/1e6:.2f} MB)")

# Enhance true color WMS image
enhance_and_save(true_color_path, "pedra_santa_eulalia_wms_true_color.png", "pedra_santa_eulalia_wms_true_color_compressed.png")

# Enhance false color WMS image
enhance_and_save(false_color_path, "pedra_santa_eulalia_wms_false_color.png", "pedra_santa_eulalia_wms_false_color_compressed.png")
