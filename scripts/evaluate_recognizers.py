#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
import torch
import jiwer
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Ground truth mapping based on previous analysis
# Keys are crop indices (1 to 12) mapping to line_00X.jpg
GT_MAPPING = {
    1: "TGL 7 Disbursed account to only go to",
    2: "This linked account will be used",
    3: "for repayment also.",
    4: "ADD SUBTASKS -> THEN CALCULATE",
    5: "(Story points)",
    7: "Complexity = what logic you are",
    8: "-> Computation applying -> No simple",
    10: "-> automation being done",
    12: "filter response & pick."
}

def load_trocr(model_name, device):
    print(f"Loading {model_name}...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def predict(processor, model, device, pil_image):
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=200)
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
    if len(image_paths) != 12:
        print(f"Expected 12 crops, found {len(image_paths)}")
    
    images = []
    for p in image_paths:
        idx = int(p.stem.split("_")[1])
        images.append((idx, p, Image.open(p).convert("RGB")))
        
    results = {
        "microsoft/trocr-base-handwritten": {"preds": {}, "time": 0.0, "cer": 0.0, "wer": 0.0},
        "microsoft/trocr-large-handwritten": {"preds": {}, "time": 0.0, "cer": 0.0, "wer": 0.0}
    }
    
    # Run evaluation
    for model_name in results.keys():
        processor, model = load_trocr(model_name, device)
        
        preds_subset = []
        gts_subset = []
        
        start_time = time.time()
        for idx, path, img in images:
            pred = predict(processor, model, device, img)
            results[model_name]["preds"][idx] = pred
            
            if idx in GT_MAPPING:
                preds_subset.append(pred)
                gts_subset.append(GT_MAPPING[idx])
                
        results[model_name]["time"] = time.time() - start_time
        
        if preds_subset:
            cer, wer = calculate_metrics(preds_subset, gts_subset)
            results[model_name]["cer"] = cer
            results[model_name]["wer"] = wer

    # Generate JSON
    json_path = output_dir / "recognizer_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate TXT report
    txt_path = output_dir / "recognizer_comparison.txt"
    with open(txt_path, "w") as f:
        f.write("PHASE 2 - RECOGNIZER BENCHMARK\n")
        f.write("==============================\n\n")
        for model_name, res in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"Total inference time (12 lines): {res['time']:.2f}s\n")
            f.write(f"CER (on 9 labeled lines): {res['cer']:.2%}\n")
            f.write(f"WER (on 9 labeled lines): {res['wer']:.2%}\n\n")
            
        f.write("LINE-BY-LINE COMPARISON\n")
        f.write("-----------------------\n")
        for idx, path, img in images:
            gt = GT_MAPPING.get(idx, "[No GT available]")
            f.write(f"Line {idx}:\n")
            f.write(f"  GT:    {gt}\n")
            f.write(f"  Base:  {results['microsoft/trocr-base-handwritten']['preds'][idx]}\n")
            f.write(f"  Large: {results['microsoft/trocr-large-handwritten']['preds'][idx]}\n\n")

    # Generate Contact Sheet
    print("Generating contact sheet...")
    # Setup fonts
    try:
        font = ImageFont.truetype("Arial", 16)
        bold_font = ImageFont.truetype("Arial Bold", 16)
    except IOError:
        font = ImageFont.load_default()
        bold_font = ImageFont.load_default()

    # Determine canvas size
    row_height = 140
    header_height = 50
    canvas_w = 1600
    canvas_h = header_height + len(images) * row_height
    
    contact = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(contact)
    
    # Draw headers
    draw.text((10, 10), "Crop Image", fill="black", font=bold_font)
    draw.text((800, 10), "TrOCR-Base", fill="black", font=bold_font)
    draw.text((1200, 10), "TrOCR-Large", fill="black", font=bold_font)
    draw.line([(0, header_height-1), (canvas_w, header_height-1)], fill="black", width=2)
    
    y = header_height
    for idx, path, img in images:
        # Resize image to fit nicely in 750x120 max
        w, h = img.size
        ratio = min(750/w, 120/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img_resized = img.resize((new_w, new_h))
        
        # Paste image centered in its cell vertically
        y_offset = y + (row_height - new_h) // 2
        contact.paste(img_resized, (10, y_offset))
        
        base_pred = results["microsoft/trocr-base-handwritten"]["preds"][idx]
        large_pred = results["microsoft/trocr-large-handwritten"]["preds"][idx]
        gt_text = GT_MAPPING.get(idx, "[No GT]")
        
        # Draw texts
        draw.text((800, y + 20), f"GT: {gt_text}", fill="green")
        draw.text((800, y + 60), f"Pred: {base_pred}", fill="blue")
        
        draw.text((1200, y + 20), f"GT: {gt_text}", fill="green")
        draw.text((1200, y + 60), f"Pred: {large_pred}", fill="blue")
        
        draw.line([(0, y + row_height - 1), (canvas_w, y + row_height - 1)], fill="gray")
        y += row_height

    contact_path = output_dir / "recognizer_comparison_contact_sheet.jpg"
    contact.save(contact_path)
    print(f"Saved {contact_path}")
    print("Done!")

if __name__ == "__main__":
    main()
