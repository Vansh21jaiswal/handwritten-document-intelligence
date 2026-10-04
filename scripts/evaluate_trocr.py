import sys
import os
import csv
import torch
import jiwer
from PIL import Image
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model.to(device)
    model.eval()

    print("Loading verified dataset...")
    csv_path = "data/adaptation/labels.csv"
    records = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "VERIFIED":
                    records.append(row)
    
    if not records:
        print("No VERIFIED records found in labels.csv")
        return

    print(f"Found {len(records)} verified records. Starting inference...")
    
    results = []
    references = []
    predictions = []
    
    for row in records:
        img_name = row['image']
        gt_text = row['text']
        img_path = f"data/adaptation/images/{img_name}"
        
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=100)
                
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            results.append((img_name, gt_text, pred_text, img_path))
            references.append(gt_text)
            predictions.append(pred_text)
            print(f"Processed {img_name}: {pred_text}")
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")

    # Calculate metrics
    print("Calculating metrics...")
    cer = jiwer.cer(references, predictions)
    wer = jiwer.wer(references, predictions)
    
    print(f"\n--- TrOCR Results ---")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")

    # Generate Markdown Report
    os.makedirs("outputs/adaptation", exist_ok=True)
    report_path = "outputs/adaptation/trocr_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TrOCR Pretrained Evaluation Results\n\n")
        f.write(f"**Baseline CNN-BiLSTM-CTC:**\nCER = 0.9432\nWER = 1.1667\n\n")
        f.write(f"**TrOCR (microsoft/trocr-base-handwritten):**\nCER = {cer:.4f}\nWER = {wer:.4f}\n\n")
        
        f.write("## Detailed Predictions\n\n")
        f.write("| Filename | Ground Truth | TrOCR Prediction | Readable? |\n")
        f.write("|----------|--------------|------------------|-----------|\n")
        
        for fname, gt, pred, _ in results:
            # simple heuristic for readability: if length is roughly similar and not totally scrambled
            # For human eyes, we will just mark '?' and let the user look at it, or just use CER < 0.3 as "Yes"
            sample_cer = jiwer.cer([gt], [pred])
            readable = "Yes" if sample_cer < 0.2 else ("Partial" if sample_cer < 0.5 else "No")
            f.write(f"| {fname} | `{gt}` | `{pred}` | {readable} (CER: {sample_cer:.2f}) |\n")
            
    print(f"Saved report to {report_path}")

    # Generate Contact Sheet
    print("Generating contact sheet...")
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 1.5 * len(results)))
    
    # If there's only 1 image, axes is not a list
    if len(results) == 1:
        axes = [axes]
        
    for i, (fname, gt, pred, img_path) in enumerate(results):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        title_color = "green" if jiwer.cer([gt], [pred]) < 0.2 else ("orange" if jiwer.cer([gt], [pred]) < 0.5 else "red")
        axes[i].set_title(f"[{fname}]\nGT: {gt}\nTrOCR: {pred}", fontsize=14, loc='left', color=title_color)
        
    plt.tight_layout()
    contact_sheet_path = "outputs/adaptation/trocr_contact_sheet.jpg"
    plt.savefig(contact_sheet_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved contact sheet to {contact_sheet_path}")

if __name__ == "__main__":
    main()
