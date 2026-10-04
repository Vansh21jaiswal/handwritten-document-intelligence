import os
import csv
import torch
import cv2
import numpy as np
import jiwer
import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def calculate_cer(pred, true):
    if not true: return 0.0
    if not pred: return 1.0
    return jiwer.cer([true], [pred])

def calculate_wer(pred, true):
    if not true: return 0.0
    if not pred: return 1.0
    return jiwer.wer([true], [pred])

class NewPreprocessor:
    def __init__(self, target_height=384):
        self.target_height = target_height
        
    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # 5. Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Illumination normalization (CLAHE) & 6. Contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        norm = clahe.apply(gray)
        
        # 4. Notebook ruling-line suppression
        # Detect horizontal lines
        binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # Dilate to cover line edges
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        lines_dilated = cv2.dilate(detected_lines, dilate_kernel, iterations=1)
        
        # Inpaint to remove lines
        cleaned = cv2.inpaint(norm, lines_dilated, 3, cv2.INPAINT_TELEA)
        
        # 1. Perspective correction (Deskew)
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            if abs(angle) < 10 and abs(angle) > 0.5:
                (h, w) = cleaned.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cleaned = cv2.warpAffine(cleaned, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
                
        # 2. Margin/background removal (Crop to text bounding box)
        binary_clean = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        coords_clean = np.column_stack(np.where(binary_clean > 0))
        if len(coords_clean) > 0:
            y_min, x_min = coords_clean.min(axis=0)
            y_max, x_max = coords_clean.max(axis=0)
            
            # Add small padding
            pad = 10
            y_min = max(0, y_min - pad)
            x_min = max(0, x_min - pad)
            y_max = min(cleaned.shape[0], y_max + pad)
            x_max = min(cleaned.shape[1], x_max + pad)
            
            cleaned = cleaned[y_min:y_max, x_min:x_max]
            
        # 7. Resize to expected height preserving aspect ratio
        h, w = cleaned.shape[:2]
        if h > 0 and w > 0:
            scale = self.target_height / float(h)
            new_w = int(w * scale)
            cleaned = cv2.resize(cleaned, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)
            
        # Convert back to RGB for TrOCR
        rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

def current_preprocessing(image_path):
    # This simulates the CRNN baseline preprocessing
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 128 / float(h)
    new_w = int(w * scale)
    resized = cv2.resize(gray, (new_w, 128))
    # Normalize and convert to PIL RGB
    norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the 39 English lines
    csv_path = "data/adaptation/labels.csv"
    english_records = []
    cpp_lines = {'line_081', 'line_083', 'line_085', 'line_086', 'line_087', 'line_089', 'line_090'}
    severe_scribbles = {'line_022'}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "VERIFIED" and row.get("line_id") not in cpp_lines and row.get("line_id") not in severe_scribbles:
                english_records.append(row)
                
    english_records = sorted(english_records, key=lambda x: x["line_id"])
    
    # Take a representative subset for this specific experiment to keep contact sheet manageable and fast
    # We will just evaluate all 39 to be statistically robust.
    
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model.to(device)
    model.eval()
    
    new_prep = NewPreprocessor(target_height=384)
    
    out_dir = "outputs/evaluation/preprocessing_experiment"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "processed_images"), exist_ok=True)
    
    results = []
    
    print("Running Preprocessing & Inference...")
    for i, row in enumerate(english_records):
        img_path = f"data/adaptation/images/{row['image_path']}"
        gt_text = row['text']
        
        # A. Raw
        img_raw = Image.open(img_path).convert("RGB")
        
        # B. Current
        img_cur = current_preprocessing(img_path)
        
        # C. New
        img_new = new_prep.process(img_path)
        
        # Save processed for visual inspection
        img_new.save(os.path.join(out_dir, "processed_images", f"new_{row['line_id']}.jpg"))
        
        preds = {}
        for name, im in [("raw", img_raw), ("current", img_cur), ("new", img_new)]:
            pixel_values = processor(images=im, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=100)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            preds[name] = pred
            
        results.append({
            "id": row["line_id"],
            "img_raw": img_raw,
            "img_cur": img_cur,
            "img_new": img_new,
            "gt": gt_text,
            "pred_raw": preds["raw"],
            "pred_cur": preds["current"],
            "pred_new": preds["new"]
        })
        
        print(f"Processed {row['line_id']} ({i+1}/{len(english_records)})")
        
    # Compute Metrics
    def agg_metrics(key):
        gts = [r["gt"] for r in results]
        prs = [r[key] for r in results]
        return jiwer.cer(gts, prs), jiwer.wer(gts, prs)
        
    cer_raw, wer_raw = agg_metrics("pred_raw")
    cer_cur, wer_cur = agg_metrics("pred_cur")
    cer_new, wer_new = agg_metrics("pred_new")
    
    # Report
    report_path = os.path.join(out_dir, "preprocessing_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TrOCR Preprocessing Experiment\n\n")
        f.write("## Aggregate Metrics\n")
        f.write(f"- **A. Raw Image:** CER = {cer_raw:.4f} | WER = {wer_raw:.4f}\n")
        f.write(f"- **B. Current Prep (CRNN):** CER = {cer_cur:.4f} | WER = {wer_cur:.4f}\n")
        f.write(f"- **C. New Prep (Advanced):** CER = {cer_new:.4f} | WER = {wer_new:.4f}\n\n")
        
        f.write("## Detailed Results\n")
        f.write("| Line ID | Ground Truth | Raw Pred | Current Pred | New Pred |\n")
        f.write("|---------|--------------|----------|--------------|----------|\n")
        for r in results:
            f.write(f"| {r['id']} | `{r['gt']}` | `{r['pred_raw']}` | `{r['pred_cur']}` | `{r['pred_new']}` |\n")

    # Contact Sheet (Top 10 samples to prevent massive images)
    sample_res = results[:12]
    fig, axes = plt.subplots(len(sample_res), 3, figsize=(15, 3 * len(sample_res)))
    
    for i, r in enumerate(sample_res):
        for j, (name, img_key, pred_key) in enumerate([("Raw", "img_raw", "pred_raw"), 
                                                     ("Current", "img_cur", "pred_cur"), 
                                                     ("New", "img_new", "pred_new")]):
            ax = axes[i, j]
            ax.imshow(np.array(r[img_key]))
            ax.axis("off")
            
            c = "green" if calculate_cer(r[pred_key], r["gt"]) < 0.2 else "red"
            title = f"[{name}]\nPred: {r[pred_key]}\nGT: {r['gt']}"
            ax.set_title(title, fontsize=10, loc='left', color=c, family='monospace')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "preprocessing_contact_sheet.jpg"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Experiment complete!")

if __name__ == "__main__":
    main()
