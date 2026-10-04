import sys
import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.training.checkpoint import load_checkpoint, rebuild_tokenizer_from_checkpoint, save_checkpoint
from src.dataset.adaptation_dataset import AdaptationDataset
from src.preprocessing.image_transforms import ImagePreprocessor
from src.models.crnn import CRNN
from src.inference.pipeline import HTRPipeline

def calculate_cer(pred, true):
    if len(true) == 0: return 0.0
    d = torch.zeros(len(true)+1, len(pred)+1)
    for i in range(len(true)+1): d[i,0] = i
    for j in range(len(pred)+1): d[0,j] = j
    for i in range(1, len(true)+1):
        for j in range(1, len(pred)+1):
            if true[i-1] == pred[j-1]: cost = 0
            else: cost = 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[-1,-1].item() / len(true)

def calculate_wer(pred, true):
    pred_w = pred.split()
    true_w = true.split()
    if len(true_w) == 0: return 0.0
    d = torch.zeros(len(true_w)+1, len(pred_w)+1)
    for i in range(len(true_w)+1): d[i,0] = i
    for j in range(len(pred_w)+1): d[0,j] = j
    for i in range(1, len(true_w)+1):
        for j in range(1, len(pred_w)+1):
            if true_w[i-1] == pred_w[j-1]: cost = 0
            else: cost = 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[-1,-1].item() / len(true_w)

def main():
    start_time = time.time()
    
    records = []
    with open("data/adaptation/labels.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "VERIFIED":
                records.append(row)
                
    random.seed(42)
    random.shuffle(records)
    train_records = records[:32]
    test_records = records[32:40]
    
    with open("data/adaptation/train.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","text","source_page","status"])
        writer.writeheader()
        writer.writerows(train_records)
        
    with open("data/adaptation/test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","text","source_page","status"])
        writer.writeheader()
        writer.writerows(test_records)
        
    print(f"Split completed: {len(train_records)} train, {len(test_records)} test.")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n--- Baseline Evaluation ---")
    pipeline = HTRPipeline("checkpoints/best.pt", config_path="configs/adaptation.yaml", device=str(device))
    
    baseline_results = []
    base_cer_sum = 0
    base_wer_sum = 0
    
    for r in test_records:
        img_path = f"data/adaptation/images/{r['image']}"
        pred = pipeline.predict(img_path)
        true = r["text"]
        cer = calculate_cer(pred, true)
        wer = calculate_wer(pred, true)
        base_cer_sum += cer
        base_wer_sum += wer
        baseline_results.append((r['image'], true, pred))
        
    base_cer = base_cer_sum / len(test_records)
    base_wer = base_wer_sum / len(test_records)
    print(f"Baseline CER: {base_cer:.4f}")
    print(f"Baseline WER: {base_wer:.4f}")
    
    print("\n--- Fine-tuning ---")
    state = load_checkpoint("checkpoints/best.pt", device=device)
    tokenizer = rebuild_tokenizer_from_checkpoint(state)
    
    model = CRNN.from_config(tokenizer.vocab_size, state["model_cfg"])
    model.load_state_dict(state["model_state"])
    model.to(device)
    
    import yaml
    with open("configs/adaptation.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)
    config = full_cfg["preprocessing"]
    
    config["domain_augmentation"] = {"enabled": True, "probability": 0.5}
    config["augmentation"] = {
        "enabled": True,
        "rotation_degrees": 2,
        "scale_limit": 0.05,
        "color_jitter": {"brightness": 0.2, "contrast": 0.2}
    }
    prep_train = ImagePreprocessor.from_config(config, augment=True)
    prep_test = ImagePreprocessor.from_config(config, augment=False)
    
    train_ds = AdaptationDataset("data/adaptation/train.csv", prep_train, tokenizer)
    test_ds = AdaptationDataset("data/adaptation/test.csv", prep_test, tokenizer)
    
    from torch.utils.data import DataLoader
    def collate_fn(batch):
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

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_cer = float('inf')
    patience = 20
    patience_counter = 0
    epochs = 100
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            input_lengths = model.compute_input_lengths(torch.full((images.size(0),), images.shape[3], dtype=torch.long)).to(device)
            loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_cer_sum = 0
        with torch.no_grad():
            for images, _, _, texts in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(2).permute(1, 0)
                for i in range(len(texts)):
                    decoded = tokenizer.decode(preds[i].cpu().tolist())
                    val_cer_sum += calculate_cer(decoded, texts[i])
                    
        val_cer = val_cer_sum / len(test_ds)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val CER: {val_cer:.4f}")
        
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            save_checkpoint(
                ckpt_dir="checkpoints",
                filename="domain_adapted.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_cer,
                tokenizer=tokenizer,
                model_cfg=state["model_cfg"]
            )
            print("  --> Saved new best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("\n--- Adapted Evaluation ---")
    pipeline_adapted = HTRPipeline("checkpoints/domain_adapted.pt", config_path="configs/adaptation.yaml", device=str(device))
    
    adapted_results = []
    adapt_cer_sum = 0
    adapt_wer_sum = 0
    
    for r in test_records:
        img_path = f"data/adaptation/images/{r['image']}"
        pred = pipeline_adapted.predict(img_path)
        true = r["text"]
        cer = calculate_cer(pred, true)
        wer = calculate_wer(pred, true)
        adapt_cer_sum += cer
        adapt_wer_sum += wer
        adapted_results.append((r['image'], true, pred))
        
    adapt_cer = adapt_cer_sum / len(test_records)
    adapt_wer = adapt_wer_sum / len(test_records)
    print(f"Adapted CER: {adapt_cer:.4f}")
    print(f"Adapted WER: {adapt_wer:.4f}")
    
    report_path = "outputs/adaptation/baseline_vs_adapted.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Domain Adaptation Results\n\n")
        f.write(f"**Baseline:**\nCER = {base_cer:.4f}\nWER = {base_wer:.4f}\n\n")
        f.write(f"**Adapted:**\nCER = {adapt_cer:.4f}\nWER = {adapt_wer:.4f}\n\n")
        abs_cer = base_cer - adapt_cer
        abs_wer = base_wer - adapt_wer
        f.write(f"**Absolute CER change:** {-abs_cer:+.4f} ({'Improved' if abs_cer > 0 else 'Degraded'})\n")
        f.write(f"**Absolute WER change:** {-abs_wer:+.4f} ({'Improved' if abs_wer > 0 else 'Degraded'})\n\n")
        
        f.write("## Comparison\n\n")
        f.write("| Sample | Ground Truth | Baseline Prediction | Adapted Prediction |\n")
        f.write("|--------|--------------|---------------------|--------------------|\n")
        
        for i in range(len(test_records)):
            sample = baseline_results[i][0]
            gt = baseline_results[i][1]
            b_pred = baseline_results[i][2]
            a_pred = adapted_results[i][2]
            f.write(f"| {sample} | `{gt}` | `{b_pred}` | `{a_pred}` |\n")
            
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f}s")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
