import sys
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.pipeline import HTRPipeline

def ruling_suppression(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(cv2.bilateralFilter(gray, 9, 75, 75), 
                                   255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    target_width = img.shape[1]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, target_width // 10), 1))
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    pure_rules = cv2.subtract(rules, strokes)
    return cv2.inpaint(img, pure_rules, 3, cv2.INPAINT_TELEA)

def contrast_normalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    bg = np.maximum(bg, 1).astype(np.float32)
    norm = cv2.divide(gray.astype(np.float32), bg) * 255
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

def combined(img):
    norm = contrast_normalization(img)
    return ruling_suppression(norm)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def main():
    pipeline = HTRPipeline(checkpoint_path="checkpoints/best.pt", device="cpu")
    
    crop_files = sorted(glob.glob("outputs/segmentation/line_*.jpg"))
    import re
    crop_files = [cf for cf in crop_files if re.match(r'.*line_\d{3}\.jpg', cf)]
    if not crop_files: return
    
    results = {}
    
    print("=== Preprocessing Ablation Inference ===")
    for cf in crop_files:
        img_orig = cv2.imread(cf)
        
        img_A = img_orig
        img_B = ruling_suppression(img_orig)
        img_C = contrast_normalization(img_orig)
        img_D = combined(img_orig)
        
        pred_A = pipeline.predict(cv2_to_pil(img_A))
        pred_B = pipeline.predict(cv2_to_pil(img_B))
        pred_C = pipeline.predict(cv2_to_pil(img_C))
        pred_D = pipeline.predict(cv2_to_pil(img_D))
        
        results[Path(cf).name] = {
            'A': (img_A, pred_A),
            'B': (img_B, pred_B),
            'C': (img_C, pred_C),
            'D': (img_D, pred_D)
        }
        print(f"Processed {Path(cf).name}")
        
    # Save Markdown Table
    md_path = Path("outputs/segmentation/preprocessing_ablation.md")
    with open(md_path, "w") as f:
        f.write("# Preprocessing Ablation Results\n\n")
        f.write("| Crop | Baseline (A) | Ruling Suppressed (B) | Normalized (C) | Combined (D) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cf in sorted(results.keys()):
            pA = results[cf]['A'][1].replace('|', '\\|')
            pB = results[cf]['B'][1].replace('|', '\\|')
            pC = results[cf]['C'][1].replace('|', '\\|')
            pD = results[cf]['D'][1].replace('|', '\\|')
            f.write(f"| {cf} | `{pA}` | `{pB}` | `{pC}` | `{pD}` |\n")
            
    print(f"\nSaved markdown table to {md_path}")
    
    # Visualization for representative crops
    # Let's pick 4 crops: e.g. line_000, line_003, line_006, line_009
    rep_keys = [k for i, k in enumerate(sorted(results.keys())) if i in [0, 3, 6, 9]]
    if not rep_keys: rep_keys = list(results.keys())[:4]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    
    col_width = 800
    row_height = 150
    margin = 20
    
    canvas_w = (col_width + margin) * 4 + margin
    canvas_h = (row_height + margin) * len(rep_keys) + margin + 40
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    headers = ["Variant A (Baseline)", "Variant B (Ruling)", "Variant C (Norm)", "Variant D (Combined)"]
    for i, h_text in enumerate(headers):
        cv2.putText(canvas, h_text, (margin + i*(col_width+margin), margin + 20), font, 0.8, (0,0,0), 2)
        
    current_y = margin + 50
    for cf in rep_keys:
        for i, var in enumerate(['A', 'B', 'C', 'D']):
            img, pred = results[cf][var]
            
            # Resize image to fit in col_width if too large
            h, w = img.shape[:2]
            scale = min(1.0, col_width / float(w))
            scale = min(scale, 80 / float(h))
            nh, nw = int(h * scale), int(w * scale)
            r_img = cv2.resize(img, (nw, nh))
            
            start_x = margin + i*(col_width+margin)
            canvas[current_y:current_y+nh, start_x:start_x+nw] = r_img
            
            # Text below
            cv2.putText(canvas, pred, (start_x, current_y + nh + 25), font, font_scale, (0,0,255), font_thickness)
            
        current_y += row_height + margin
        
    vis_path = "outputs/segmentation/preprocessing_ablation.jpg"
    cv2.imwrite(vis_path, canvas)
    print(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    main()
