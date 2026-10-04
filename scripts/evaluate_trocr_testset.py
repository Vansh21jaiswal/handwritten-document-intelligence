import os
import csv
import torch
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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test split
    csv_path = "data/adaptation/test_split.csv"
    test_records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_records.append(row)
            
    print(f"Loaded {len(test_records)} test samples.")
    
    # Load Model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model.to(device)
    model.eval()
    
    results = []
    preds_all = []
    truths_all = []
    
    os.makedirs("outputs/evaluation", exist_ok=True)
    
    print("Running Inference...")
    for row in test_records:
        img_name = row['image_path']
        gt_text = row['text']
        img_path = f"data/adaptation/images/{img_name}"
        
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=100)
            
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        cer = calculate_cer(pred_text, gt_text)
        wer = calculate_wer(pred_text, gt_text)
        
        results.append({
            "id": row["line_id"],
            "image_path": img_path,
            "gt": gt_text,
            "pred": pred_text,
            "cer": cer,
            "wer": wer
        })
        preds_all.append(pred_text)
        truths_all.append(gt_text)
        print(f"{row['line_id']} -> CER: {cer:.2f}, WER: {wer:.2f}")

    total_cer = jiwer.cer(truths_all, preds_all)
    total_wer = jiwer.wer(truths_all, preds_all)
    
    print(f"\nAggregate CER: {total_cer:.4f}")
    print(f"Aggregate WER: {total_wer:.4f}")
    
    # Generate Report
    report_path = "outputs/evaluation/trocr_test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TrOCR Zero-Shot Evaluation Report\n\n")
        f.write("## Aggregate Metrics\n")
        f.write(f"- **TrOCR CER:** {total_cer:.4f}\n")
        f.write(f"- **TrOCR WER:** {total_wer:.4f}\n\n")
        
        f.write("## Comparison against Baselines\n")
        f.write("- **CNN-BiLSTM-CTC Baseline:** CER = 0.9514 | WER = 1.0667\n")
        f.write("- **Adapted CNN-BiLSTM-CTC:** CER = 0.9011 | WER = 1.0000\n\n")
        
        f.write("## Per-Sample Results\n")
        f.write("| Line ID | Ground Truth | TrOCR Prediction | CER | WER |\n")
        f.write("|---------|--------------|------------------|-----|-----|\n")
        for r in results:
            f.write(f"| {r['id']} | `{r['gt']}` | `{r['pred']}` | {r['cer']:.4f} | {r['wer']:.4f} |\n")
            
    # Generate Contact Sheet using matplotlib
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 2 * len(results)))
    if len(results) == 1: axes = [axes]
    
    for i, r in enumerate(results):
        img = Image.open(r["image_path"])
        axes[i].imshow(img)
        axes[i].axis("off")
        
        color = "green" if r["cer"] < 0.2 else ("orange" if r["cer"] < 0.5 else "red")
        
        title = f"ID: {r['id']}\nGT:   {r['gt']}\nPred: {r['pred']}\n(CER: {r['cer']:.2f}, WER: {r['wer']:.2f})"
        axes[i].set_title(title, fontsize=12, loc='left', color=color, family='monospace')
        
    plt.tight_layout()
    plt.savefig("outputs/evaluation/trocr_test_contact_sheet.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved report to {report_path}")
    print("Saved contact sheet to outputs/evaluation/trocr_test_contact_sheet.jpg")

if __name__ == "__main__":
    main()
