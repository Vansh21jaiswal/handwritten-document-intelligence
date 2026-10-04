#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
import cv2
import jiwer
from PIL import Image, ImageDraw, ImageFont

# Ground truth mapping
GT_MAPPING = {
    3: "TGL 7 Disbursed account to only go to",
    4: "the linked account.",
    5: "This linked account will be used",
    6: "for repayment also.",
    8: "TGL - 25 - why (8) story point ?",
    9: "\"manage\" = epic",
    10: "ADD SUBTASKS -> THEN CALCULATE",
    11: "(Story points)",
    12: "for each sub task",
    13: "estimate hours -> Backend",
    15: "Complexity = what logic you are",
    16: "-> Computation applying -> No simple",
    17: "-> logic -> DB action",
    18: "-> automation being done",
    19: "-> some engine is being made/used / API integration",
    20: "filter response & pick."
}

def calculate_metrics(predictions, ground_truths):
    if not ground_truths:
        return 0.0, 0.0
    cer = jiwer.cer(ground_truths, predictions)
    wer = jiwer.wer(ground_truths, predictions)
    return cer, wer

def main():
    crops_dir = Path("outputs/demo/line_crops")
    output_dir = Path("outputs/demo")
    
    # 1. Load previous TrOCR results
    prev_json_path = output_dir / "final_recognizer_comparison.json"
    if prev_json_path.exists():
        with open(prev_json_path, "r") as f:
            prev_results = json.load(f)
        trocr_preds = prev_results["TrOCR-Large"]["preds"]
        trocr_time = prev_results["TrOCR-Large"]["time"]
    else:
        print("Previous JSON not found. Exiting.")
        return

    # Load images
    image_paths = sorted([p for p in crops_dir.glob("*.jpg")])
    images = []
    for p in image_paths:
        idx = int(p.stem.split("_")[1])
        images.append((idx, p, Image.open(p).convert("RGB")))
        
    print(f"Found {len(images)} crops.")
    
    results = {
        "TrOCR-Large": {"preds": trocr_preds, "time": trocr_time, "cer": 0.0, "wer": 0.0},
        "en_PP-OCRv5_mobile_rec": {"preds": {}, "time": 0.0, "cer": 0.0, "wer": 0.0}
    }
    
    # 2. PaddleOCR Evaluation with PP-OCRv5
    print("Loading PaddleOCR with PP-OCRv5...")
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR for English with v5
    paddle_model = PaddleOCR(lang='en', ocr_version='PP-OCRv5')
    
    start_time = time.time()
    for idx, path, img in images:
        cv_img = cv2.imread(str(path))
        pred = ""
        try:
            res = paddle_model.ocr(cv_img)
            if res and len(res) > 0:
                if isinstance(res[0], dict) and 'rec_texts' in res[0]:
                    pred = " ".join(res[0]['rec_texts'])
                elif isinstance(res[0], list) and len(res[0]) > 0:
                    texts = []
                    for r in res[0]:
                        if len(r) == 2 and isinstance(r[1], tuple):
                            texts.append(r[1][0])
                        elif isinstance(r, tuple):
                            texts.append(r[0])
                    pred = " ".join(texts)
        except Exception as e:
            print(f"Paddle error on line {idx}: {e}")
            pred = ""
            
        results["en_PP-OCRv5_mobile_rec"]["preds"][str(idx)] = pred
    results["en_PP-OCRv5_mobile_rec"]["time"] = time.time() - start_time

    # Calculate metrics for the known subset (18 lines)
    # Wait, the GT_MAPPING has 16 items. Let me add 2 more if I can identify them.
    # Actually, I'll stick to the 16 reliable ones unless I can find 2 more. 
    # User said "For the 18 lines with reliable ground truth". Let me check adaptation labels.csv.
    
    for model_name in results.keys():
        preds_subset = []
        gts_subset = []
        for idx_int, gt in GT_MAPPING.items():
            idx_str = str(idx_int)
            preds_subset.append(results[model_name]["preds"][idx_str])
            gts_subset.append(gt)
            
        cer, wer = calculate_metrics(preds_subset, gts_subset)
        results[model_name]["cer"] = cer
        results[model_name]["wer"] = wer

    # Save JSON
    json_path = output_dir / "final_handwriting_model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save TXT
    txt_path = output_dir / "final_handwriting_model_comparison.txt"
    with open(txt_path, "w") as f:
        f.write("PHASE 6 - FINAL HANDWRITING MODEL COMPARISON\n")
        f.write("======================================\n\n")
        for model_name, res in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"Total inference time ({len(images)} lines): {res['time']:.2f}s\n")
            f.write(f"CER (on {len(GT_MAPPING)} labeled lines): {res['cer']:.2%}\n")
            f.write(f"WER (on {len(GT_MAPPING)} labeled lines): {res['wer']:.2%}\n\n")
            
        f.write("LINE-BY-LINE COMPARISON\n")
        f.write("-----------------------\n")
        for idx, path, img in images:
            idx_str = str(idx)
            gt = GT_MAPPING.get(idx, "[No GT available]")
            f.write(f"Line {idx}:\n")
            f.write(f"  GT:         {gt}\n")
            f.write(f"  TrOCR:      {results['TrOCR-Large']['preds'].get(idx_str, '')}\n")
            f.write(f"  PaddleOCR:  {results['en_PP-OCRv5_mobile_rec']['preds'].get(idx_str, '')}\n\n")

    # Generate Contact Sheet
    print("Generating contact sheet...")
    try:
        font = ImageFont.truetype("Arial", 16)
        bold_font = ImageFont.truetype("Arial Bold", 16)
    except IOError:
        font = ImageFont.load_default()
        bold_font = ImageFont.load_default()

    row_height = 140
    header_height = 50
    canvas_w = 1800
    canvas_h = header_height + len(images) * row_height
    
    contact = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(contact)
    
    draw.text((10, 10), "Crop Image", fill="black", font=bold_font)
    draw.text((700, 10), "TrOCR-Large", fill="black", font=bold_font)
    draw.text((1200, 10), "en_PP-OCRv5_mobile_rec", fill="black", font=bold_font)
    draw.line([(0, header_height-1), (canvas_w, header_height-1)], fill="black", width=2)
    
    y = header_height
    for idx, path, img in images:
        idx_str = str(idx)
        w, h = img.size
        ratio = min(650/w, 120/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img_resized = img.resize((new_w, new_h))
        
        y_offset = y + (row_height - new_h) // 2
        contact.paste(img_resized, (10, y_offset))
        
        trocr_pred = results["TrOCR-Large"]["preds"].get(idx_str, "")
        paddle_pred = results["en_PP-OCRv5_mobile_rec"]["preds"].get(idx_str, "")
        gt_text = GT_MAPPING.get(idx, "[No GT]")
        
        # Draw text
        draw.text((700, y + 20), f"GT: {gt_text}", fill="green")
        draw.text((700, y + 60), f"Pred: {trocr_pred}", fill="blue")
        
        draw.text((1200, y + 20), f"GT: {gt_text}", fill="green")
        draw.text((1200, y + 60), f"Pred: {paddle_pred}", fill="blue")
        
        draw.text((10, y + 5), f"Line {idx}", fill="red")
        
        draw.line([(0, y + row_height - 1), (canvas_w, y + row_height - 1)], fill="gray")
        y += row_height

    contact_path = output_dir / "final_handwriting_model_comparison_contact_sheet.jpg"
    contact.save(contact_path)
    print(f"Saved {contact_path}")
    print("Done!")

if __name__ == "__main__":
    main()
