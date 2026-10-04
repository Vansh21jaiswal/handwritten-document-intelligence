import sys
import os
import csv
import random
import time
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.crnn import CRNN
from src.dataset.adaptation_dataset import AdaptationDataset
from src.preprocessing.image_transforms import ImagePreprocessor
import jiwer
from src.training.checkpoint import load_checkpoint, save_checkpoint, rebuild_tokenizer_from_checkpoint
from src.inference.pipeline import HTRPipeline

def calculate_cer(pred, true):
    if not true: return 0.0
    if not pred: return 1.0
    return jiwer.cer([true], [pred])

def calculate_wer(pred, true):
    if not true: return 0.0
    if not pred: return 1.0
    return jiwer.wer([true], [pred])

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def collate_fn(batch, tokenizer):
    max_w = max(item["image"].shape[2] for item in batch)
    padded_images = []
    for item in batch:
        img = item["image"]
        if img.shape[2] < max_w:
            pad = torch.ones(1, img.shape[1], max_w - img.shape[2])
            img = torch.cat([img, pad], dim=2)
        padded_images.append(img)
        
    images = torch.stack(padded_images)
    texts = [item["text"] for item in batch]
    encoded = [tokenizer.encode(t) for t in texts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    max_target_len = target_lengths.max().item() if len(target_lengths) > 0 else 0
    targets = torch.zeros((len(batch), max_target_len), dtype=torch.long)
    for i, e in enumerate(encoded):
        targets[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return images, targets, target_lengths, texts

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Filter and Split Data
    csv_path = "data/adaptation/labels.csv"
    verified_records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "VERIFIED":
                verified_records.append(row)
                
    if len(verified_records) != 47:
        print(f"WARNING: Expected 47 verified records, found {len(verified_records)}")
        
    random.shuffle(verified_records)
    
    train_records = verified_records[:32]
    val_records = verified_records[32:37]
    test_records = verified_records[37:47]
    
    # Write splits
    split_dir = Path("data/adaptation")
    for name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        with open(split_dir / f"{name}_split.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["line_id", "image_path", "status", "text"])
            writer.writeheader()
            writer.writerows(records)
            
    print(f"Splits saved: {len(train_records)} Train, {len(val_records)} Val, {len(test_records)} Test")
    print("Test IDs:", [r["line_id"] for r in test_records])
    
    # Load state and tokenizer
    state = load_checkpoint("checkpoints/best.pt", device=device)
    tokenizer = rebuild_tokenizer_from_checkpoint(state)
    
    # Setup Preprocessors
    with open("configs/adaptation.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)
    config = full_cfg["preprocessing"]
    
    # Force augmentation settings for training
    config["domain_augmentation"] = {"enabled": True, "probability": 0.8}
    config["augmentation"] = {
        "enabled": True,
        "rotation_degrees": 2,
        "scale_limit": 0.05,
        "color_jitter": {"brightness": 0.2, "contrast": 0.2}
    }
    
    prep_train = ImagePreprocessor.from_config(config, augment=True)
    prep_val = ImagePreprocessor.from_config(config, augment=False)
    
    train_ds = AdaptationDataset(str(split_dir / "train_split.csv"), prep_train, tokenizer)
    val_ds = AdaptationDataset(str(split_dir / "val_split.csv"), prep_val, tokenizer)
    test_ds = AdaptationDataset(str(split_dir / "test_split.csv"), prep_val, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    # --- BASELINE EVALUATION ---
    print("\n--- Baseline Evaluation on Test Set ---")
    pipeline = HTRPipeline("checkpoints/best.pt", config_path="configs/adaptation.yaml", device=str(device))
    baseline_results = []
    b_cer_sum, b_wer_sum = 0, 0
    for r in test_records:
        img_path = f"data/adaptation/images/{r['image_path']}"
        pred = pipeline.predict(img_path)
        true = r["text"]
        cer = calculate_cer(pred, true)
        wer = calculate_wer(pred, true)
        b_cer_sum += cer
        b_wer_sum += wer
        baseline_results.append({
            "img": r['image_path'],
            "true": true,
            "pred": pred,
            "path": img_path
        })
        
    baseline_cer = b_cer_sum / len(test_records)
    baseline_wer = b_wer_sum / len(test_records)
    print(f"Baseline CER: {baseline_cer:.4f} | WER: {baseline_wer:.4f}")
    
    # --- TRAINING ---
    print("\n--- Fine-tuning (CNN Frozen) ---")
    model = CRNN.from_config(tokenizer.vocab_size, state["model_cfg"])
    model.load_state_dict(state["model_state"])
    
    # FREEZE CNN
    for param in model.cnn.parameters():
        param.requires_grad = False
        
    model.to(device)
    
    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(opt_params, lr=1e-4, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_cer = float('inf')
    best_epoch = 0
    epochs = 50
    ckpt_path = "checkpoints/adapted_v2.pt"
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            # Compute original unpadded lengths properly
            # width / 4 (width reduction)
            # Actually, collate_fn doesn't pass original widths, but passing images.size(3) works for CTC
            input_lengths = model.compute_input_lengths(torch.full((images.size(0),), images.size(3), dtype=torch.long)).to(device)
            
            loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Eval
        model.eval()
        val_cer_sum = 0
        with torch.no_grad():
            for images, _, _, texts in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(2).permute(1, 0)
                for i in range(len(texts)):
                    decoded = tokenizer.decode(preds[i].cpu().tolist())
                    val_cer_sum += calculate_cer(decoded, texts[i])
                    
        val_cer = val_cer_sum / len(val_ds)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val CER: {val_cer:.4f}")
        
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_epoch = epoch + 1
            save_checkpoint(
                ckpt_dir="checkpoints",
                filename="adapted_v2.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_cer,
                tokenizer=tokenizer,
                model_cfg=state["model_cfg"]
            )
            
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s. Best epoch: {best_epoch} (Val CER: {best_val_cer:.4f})")
    
    # --- ADAPTED EVALUATION ---
    print("\n--- Adapted Evaluation on Test Set ---")
    pipeline_adapted = HTRPipeline("checkpoints/adapted_v2.pt", config_path="configs/adaptation.yaml", device=str(device))
    adapted_results = []
    a_cer_sum, a_wer_sum = 0, 0
    for r in test_records:
        img_path = f"data/adaptation/images/{r['image_path']}"
        pred = pipeline_adapted.predict(img_path)
        true = r["text"]
        cer = calculate_cer(pred, true)
        wer = calculate_wer(pred, true)
        a_cer_sum += cer
        a_wer_sum += wer
        adapted_results.append({
            "img": r['image_path'],
            "pred": pred
        })
        
    adapted_cer = a_cer_sum / len(test_records)
    adapted_wer = a_wer_sum / len(test_records)
    print(f"Adapted CER: {adapted_cer:.4f} | WER: {adapted_wer:.4f}")
    
    # --- REPORT GENERATION ---
    report_path = "outputs/adaptation/experiment_v2_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Controlled Domain Adaptation Experiment\n\n")
        f.write(f"- **Training Time:** {train_time:.1f}s\n")
        f.write(f"- **Best Epoch:** {best_epoch}\n")
        f.write(f"- **Train Samples:** 32 (IDs: {', '.join([r['line_id'] for r in train_records])})\n")
        f.write(f"- **Validation Samples:** 5 (IDs: {', '.join([r['line_id'] for r in val_records])})\n")
        f.write(f"- **Test Samples:** 10 (IDs: {', '.join([r['line_id'] for r in test_records])})\n\n")
        
        f.write("## Metrics on Held-Out Test Set\n")
        f.write(f"- **Baseline CER:** {baseline_cer:.4f}\n")
        f.write(f"- **Baseline WER:** {baseline_wer:.4f}\n")
        f.write(f"- **Adapted CER:** {adapted_cer:.4f}\n")
        f.write(f"- **Adapted WER:** {adapted_wer:.4f}\n")
        f.write(f"- **Absolute CER Change:** {adapted_cer - baseline_cer:+.4f}\n")
        f.write(f"- **Absolute WER Change:** {adapted_wer - baseline_wer:+.4f}\n\n")
        
        f.write("## Prediction Comparison\n")
        f.write("| Image | Ground Truth | Baseline Prediction | Adapted Prediction |\n")
        f.write("|-------|--------------|---------------------|--------------------|\n")
        for i in range(len(test_records)):
            b = baseline_results[i]
            a = adapted_results[i]
            f.write(f"| {b['img']} | `{b['true']}` | `{b['pred']}` | `{a['pred']}` |\n")

    # --- CONTACT SHEET GENERATION ---
    print("Generating contact sheet...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 30
    
    row_data = []
    max_w = 0
    total_h = margin
    for i in range(len(test_records)):
        img = cv2.imread(baseline_results[i]["path"])
        h, w = img.shape[:2]
        max_w = max(max_w, w)
        row_h = max(h, 120) + 40
        total_h += row_h + margin
        row_data.append((img, baseline_results[i], adapted_results[i], row_h))
        
    canvas_w = max_w + 600 + margin*3
    canvas = np.ones((total_h, canvas_w, 3), dtype=np.uint8) * 255
    
    current_y = margin
    for img, b, a, row_h in row_data:
        h, w = img.shape[:2]
        canvas[current_y+20:current_y+20+h, margin:margin+w] = img
        
        text_x = margin + max_w + margin
        cv2.putText(canvas, f"ID: {b['img']}", (margin, current_y + 15), font, 0.5, (0,0,0), 1)
        
        cv2.putText(canvas, f"GT: {b['true']}", (text_x, current_y + 20), font, 0.5, (0,0,0), 1)
        
        b_color = (0,0,200) if calculate_cer(b['pred'], b['true']) > 0.2 else (0,150,0)
        cv2.putText(canvas, f"Base: {b['pred']}", (text_x, current_y + 60), font, 0.5, b_color, 1)
        
        a_color = (0,0,200) if calculate_cer(a['pred'], b['true']) > 0.2 else (0,150,0)
        cv2.putText(canvas, f"Adapt: {a['pred']}", (text_x, current_y + 100), font, 0.5, a_color, 1)
        
        current_y += row_h + margin
        
    contact_path = "outputs/adaptation/experiment_v2_contact_sheet.jpg"
    cv2.imwrite(contact_path, canvas)
    
    print(f"\nExperiment complete. Contact sheet: {contact_path}")

if __name__ == "__main__":
    main()
