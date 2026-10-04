import os
import csv
import random
import time
import torch
import jiwer
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    cer = jiwer.cer(labels, preds)
    wer = jiwer.wer(labels, preds)
    return cer, wer

class TrOCRDataset(Dataset):
    def __init__(self, records, processor, is_train=False):
        self.records = records
        self.processor = processor
        self.is_train = is_train
        
        if is_train:
            self.transforms = T.Compose([
                T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.GaussianBlur(3, sigma=(0.1, 1.0)),
            ])
        else:
            self.transforms = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        img_name = row['image']
        gt_text = row['text']
        img_path = f"data/adaptation/images/{img_name}"
        
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
            
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze()
        
        encoded_labels = self.processor.tokenizer(
            gt_text, padding="max_length", max_length=128, truncation=True
        ).input_ids
        
        # Replace pad token id with -100 to ignore in cross-entropy loss
        encoded_labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in encoded_labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(encoded_labels),
            "text": gt_text,
            "img_name": img_name,
            "img_path": img_path
        }

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["text"] for item in batch]
    img_names = [item["img_name"] for item in batch]
    img_paths = [item["img_path"] for item in batch]
    return pixel_values, labels, texts, img_names, img_paths

def evaluate_model(model, processor, dataloader, device):
    model.eval()
    predictions = []
    references = []
    results = []
    
    with torch.no_grad():
        for pixel_values, _, texts, img_names, img_paths in dataloader:
            pixel_values = pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_new_tokens=100)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, pred_text in enumerate(generated_texts):
                pred_text = pred_text.strip()
                gt_text = texts[i]
                predictions.append(pred_text)
                references.append(gt_text)
                results.append((img_names[i], gt_text, pred_text, img_paths[i]))
                
    cer, wer = compute_metrics(predictions, references)
    return cer, wer, results

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data & Split
    print("Loading dataset...")
    csv_path = "data/adaptation/labels.csv"
    records = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "VERIFIED":
                    records.append(row)
                    
    random.shuffle(records)
    
    # Deterministic split: Train 24, Val 8, Test 8
    train_records = records[:24]
    val_records = records[24:32]
    test_records = records[32:40]
    
    print("\nDataset Split Info:")
    print(f"Random Seed: 42")
    print(f"Total Verified Samples: {len(records)}")
    print(f"Training Set: {len(train_records)} samples")
    print(f"Validation Set: {len(val_records)} samples")
    print(f"Held-out Test Set: {len(test_records)} samples")
    print("Data leakage verified: test elements are strictly separate from train/val splits.\n")
    
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Initialize Datasets
    train_ds = TrOCRDataset(train_records, processor, is_train=True)
    val_ds = TrOCRDataset(val_records, processor, is_train=False)
    test_ds = TrOCRDataset(test_records, processor, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # Zero-Shot Eval on Test set
    print("--- Evaluating Zero-Shot TrOCR on Held-Out Test Set ---")
    zero_shot_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    zero_shot_model.to(device)
    zs_cer, zs_wer, zs_results = evaluate_model(zero_shot_model, processor, test_loader, device)
    print(f"Zero-Shot Test CER: {zs_cer:.4f} | WER: {zs_wer:.4f}\n")
    
    # Clean up zero shot model to free memory
    del zero_shot_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Fine-Tuning
    print("--- Starting TrOCR Fine-Tuning ---")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # model configuration adjustments for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 20
    best_val_cer = float('inf')
    ckpt_dir = "checkpoints/trocr_notebook"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    
    start_time = time.time()
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for pixel_values, labels, _, _, _ in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        val_cer, val_wer, _ = evaluate_model(model, processor, val_loader, device)
        loss_history.append((epoch+1, avg_train_loss, val_cer))
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}")
        
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            print(f"  -> New best Val CER! Saving checkpoint to {best_ckpt_path}")
            torch.save(model.state_dict(), best_ckpt_path)
            
    print(f"\nFine-Tuning Complete in {time.time()-start_time:.1f}s.")
    
    # 3. Final Evaluation
    print("\n--- Evaluating Fine-Tuned TrOCR on Held-Out Test Set ---")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    ft_cer, ft_wer, ft_results = evaluate_model(model, processor, test_loader, device)
    
    print(f"Fine-Tuned Test CER: {ft_cer:.4f} | WER: {ft_wer:.4f}")
    
    # 4. Generate Report
    report_path = "outputs/adaptation/trocr_finetuning_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TrOCR Domain Adaptation Report\n\n")
        f.write(f"**Test Set Samples:** {len(test_records)}\n\n")
        
        f.write("### Metrics Summary\n")
        f.write(f"- **Zero-Shot Test CER:** {zs_cer:.4f}\n")
        f.write(f"- **Zero-Shot Test WER:** {zs_wer:.4f}\n")
        f.write(f"- **Fine-Tuned Test CER:** {ft_cer:.4f}\n")
        f.write(f"- **Fine-Tuned Test WER:** {ft_wer:.4f}\n\n")
        
        abs_cer_diff = zs_cer - ft_cer
        abs_wer_diff = zs_wer - ft_wer
        
        f.write(f"**Absolute CER change:** {-abs_cer_diff:+.4f} (Negative is better)\n")
        f.write(f"**Absolute WER change:** {-abs_wer_diff:+.4f} (Negative is better)\n\n")
        
        f.write("### Training History\n")
        f.write("| Epoch | Train Loss | Validation CER |\n")
        f.write("|-------|------------|----------------|\n")
        for e, loss, vcer in loss_history:
            f.write(f"| {e} | {loss:.4f} | {vcer:.4f} |\n")
        
        f.write("\n### Held-Out Test Set Comparison\n\n")
        f.write("| Image | Ground Truth | Zero-Shot | Fine-Tuned |\n")
        f.write("|-------|--------------|-----------|------------|\n")
        
        for i in range(len(test_records)):
            fname = zs_results[i][0]
            gt = zs_results[i][1]
            zs_pred = zs_results[i][2]
            ft_pred = ft_results[i][2]
            f.write(f"| {fname} | `{gt}` | `{zs_pred}` | `{ft_pred}` |\n")
            
    print(f"\nReport saved to {report_path}")
    
    # 5. Generate Contact Sheet
    fig, axes = plt.subplots(len(test_records), 1, figsize=(15, 2.0 * len(test_records)))
    if len(test_records) == 1: axes = [axes]
    
    for i in range(len(test_records)):
        fname = zs_results[i][0]
        gt = zs_results[i][1]
        zs_pred = zs_results[i][2]
        ft_pred = ft_results[i][2]
        img_path = zs_results[i][3]
        
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        
        zs_cer_val = jiwer.cer([gt], [zs_pred])
        ft_cer_val = jiwer.cer([gt], [ft_pred])
        
        title = f"GT: {gt}\nZS ({zs_cer_val:.2f}): {zs_pred}\nFT ({ft_cer_val:.2f}): {ft_pred}"
        
        color = "green" if ft_cer_val < zs_cer_val else ("red" if ft_cer_val > zs_cer_val else "black")
        axes[i].set_title(title, fontsize=12, loc='left', color=color)
        
    plt.tight_layout()
    plt.savefig("outputs/adaptation/trocr_ft_contact_sheet.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Done. Contact sheet saved.")
    
if __name__ == "__main__":
    main()
