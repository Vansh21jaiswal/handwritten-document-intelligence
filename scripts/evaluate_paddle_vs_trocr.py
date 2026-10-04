#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
import torch
import jiwer
from PIL import Image, ImageDraw, ImageFont
import cv2

# Ground truth mapping based on previous analysis
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

def load_trocr(device):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    model_name = "microsoft/trocr-large-handwritten"
    print(f"Loading {model_name}...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def predict_trocr(processor, model, device, pil_image):
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(
            pixel_values, 
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def calculate_metrics(predictions, ground_truths):
    if not ground_truths:
        return 0.0, 0.0
    cer = jiwer.cer(ground_truths, predictions)
    wer = jiwer.wer(ground_truths, predictions)
    return cer, wer

def main():
    crops_dir = Path("outputs/demo/line_crops")
    output_dir = Path("outputs/demo")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load images
    image_paths = sorted([p for p in crops_dir.glob("*.jpg")])
    images = []
    for p in image_paths:
        idx = int(p.stem.split("_")[1])
        images.append((idx, p, Image.open(p).convert("RGB")))
        
    print(f"Found {len(images)} crops.")
    
    results = {
        "TrOCR-Large": {"preds": {}, "time": 0.0, "cer": 0.0, "wer": 0.0},
        "PaddleOCR": {"preds": {}, "time": 0.0, "cer": 0.0, "wer": 0.0}
    }
    
    # 1. TrOCR Evaluation
    processor, model = load_trocr(device)
    start_time = time.time()
    for idx, path, img in images:
        pred = predict_trocr(processor, model, device, img)
        results["TrOCR-Large"]["preds"][idx] = pred
    results["TrOCR-Large"]["time"] = time.time() - start_time
    
    # Free up memory
    del processor
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 2. PaddleOCR Evaluation
    print("Loading PaddleOCR...")
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR for English
    paddle_model = PaddleOCR(lang='en') 
    
    start_time = time.time()
    for idx, path, img in images:
        cv_img = cv2.imread(str(path))
        # PaddleOCR returns different formats depending on version/backend
        pred = ""
        try:
            # Full OCR on crop since det=False causes issues in v3.7+ paddlex backend
            res = paddle_model.ocr(cv_img)
            if res and len(res) > 0:
                # PaddleX dictionary format
                if isinstance(res[0], dict) and 'rec_texts' in res[0]:
                    pred = " ".join(res[0]['rec_texts'])
                # Legacy list format
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
            
        results["PaddleOCR"]["preds"][idx] = pred
    results["PaddleOCR"]["time"] = time.time() - start_time

    # Calculate metrics for the known subset
    for model_name in results.keys():
        preds_subset = []
        gts_subset = []
        for idx, gt in GT_MAPPING.items():
            preds_subset.append(results[model_name]["preds"][idx])
            gts_subset.append(gt)
            
        cer, wer = calculate_metrics(preds_subset, gts_subset)
        results[model_name]["cer"] = cer
        results[model_name]["wer"] = wer

    # Save JSON
    json_path = output_dir / "final_recognizer_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save TXT
    txt_path = output_dir / "final_recognizer_comparison.txt"
    with open(txt_path, "w") as f:
        f.write("PHASE 5 - FINAL RECOGNIZER COMPARISON\n")
        f.write("======================================\n\n")
        for model_name, res in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"Total inference time ({len(images)} lines): {res['time']:.2f}s\n")
            f.write(f"CER (on {len(GT_MAPPING)} labeled lines): {res['cer']:.2%}\n")
            f.write(f"WER (on {len(GT_MAPPING)} labeled lines): {res['wer']:.2%}\n\n")
            
        f.write("LINE-BY-LINE COMPARISON\n")
        f.write("-----------------------\n")
        for idx, path, img in images:
            gt = GT_MAPPING.get(idx, "[No GT available]")
            f.write(f"Line {idx}:\n")
            f.write(f"  GT:         {gt}\n")
            f.write(f"  TrOCR:      {results['TrOCR-Large']['preds'][idx]}\n")
            f.write(f"  PaddleOCR:  {results['PaddleOCR']['preds'][idx]}\n\n")

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
    draw.text((1200, 10), "PaddleOCR", fill="black", font=bold_font)
    draw.line([(0, header_height-1), (canvas_w, header_height-1)], fill="black", width=2)
    
    y = header_height
    for idx, path, img in images:
        w, h = img.size
        ratio = min(650/w, 120/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img_resized = img.resize((new_w, new_h))
        
        y_offset = y + (row_height - new_h) // 2
        contact.paste(img_resized, (10, y_offset))
        
        trocr_pred = results["TrOCR-Large"]["preds"][idx]
        paddle_pred = results["PaddleOCR"]["preds"][idx]
        gt_text = GT_MAPPING.get(idx, "[No GT]")
        
        # Draw text
        draw.text((700, y + 20), f"GT: {gt_text}", fill="green")
        draw.text((700, y + 60), f"Pred: {trocr_pred}", fill="blue")
        
        draw.text((1200, y + 20), f"GT: {gt_text}", fill="green")
        draw.text((1200, y + 60), f"Pred: {paddle_pred}", fill="blue")
        
        draw.text((10, y + 5), f"Line {idx}", fill="red")
        
        draw.line([(0, y + row_height - 1), (canvas_w, y + row_height - 1)], fill="gray")
        y += row_height

    contact_path = output_dir / "final_recognizer_comparison_contact_sheet.jpg"
    contact.save(contact_path)
    print(f"Saved {contact_path}")
    print("Done!")

if __name__ == "__main__":
    main()
