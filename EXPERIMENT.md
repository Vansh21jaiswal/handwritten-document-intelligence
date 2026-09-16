# Experiment Log — Handwritten Document Intelligence

This document records baseline architecture decisions, training configuration,
and observed results. Performance numbers are only added after real experiments.

---

## Baseline Architecture

**Model type:** CNN-BiLSTM-CTC (CRNN)

```
Input:  (B, 1, 128, W)   — grayscale line image, variable width

CNN Feature Extractor
  Block 0: Conv(1→32)   + BN + ReLU + MaxPool(2,2)  → ×2 H, ×2 W reduction
  Block 1: Conv(32→64)  + BN + ReLU + MaxPool(2,2)  → ×2 H, ×2 W reduction
  Block 2: Conv(64→128) + BN + ReLU + MaxPool(2,1)  → ×2 H, W unchanged
  Block 3: Conv(128→256)+ BN + ReLU + MaxPool(2,1)  → ×2 H, W unchanged

Total CNN reduction: H: ÷16 (128→8px), W: ÷4

Height Collapse
  AdaptiveAvgPool2d(1, None)  →  (B, 256, 1, W/4)
  squeeze + permute           →  (W/4, B, 256)

BiLSTM
  2 stacked bidirectional LSTM layers
  hidden_size=256 per direction → output dim = 512

Linear Classifier
  Linear(512 → vocab_size)    →  (W/4, B, vocab_size)

Output: raw logits for nn.CTCLoss
```

**Total parameters:** ~3M (varies slightly with vocabulary size)

---

## Dataset

**Source:** [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) via HuggingFace Datasets Viewer API  
**Content:** IAM Handwriting Database — English handwritten line images  
**Access:** REST API (no local download; not stored in repository)

| Split      | Samples |
|------------|---------|
| Train      | 6,482   |
| Validation | 976     |
| Test       | 1,861   |

> **Data leakage policy:** Tokenizer vocabulary is built from training texts only. Validation and test splits are never seen by the tokenizer or model during training.

---

## Preprocessing

1. Convert to grayscale (`L` mode)
2. Resize to fixed height 128 px, width scaled proportionally (aspect ratio preserved)
3. Convert to `float32` tensor in `[0, 1]` via `TF.to_tensor()`
4. Normalize: `(pixel - 0.5) / 0.5` → output in `[-1, 1]`
5. Batch: variable-width images padded to max width in batch (`pad_value=0.0`)

---

## CTC Decoding

**Method:** Greedy (argmax) decoding

```
For each timestep t:
    predicted_id[t] = argmax(logits[t])

Collapse consecutive duplicate IDs:
    [a, a, blank, b, b, blank, c]  →  [a, blank, b, blank, c]

Remove blank tokens:
    [a, blank, b, blank, c]  →  [a, b, c]

Decode IDs → string via tokenizer
```

**Blank index:** 0 (matches `nn.CTCLoss(blank=0)`)

---

## Evaluation Metrics

**CER (Character Error Rate)**
```
CER = edit_distance(reference_chars, predicted_chars) / len(reference_chars)
```

**WER (Word Error Rate)**
```
WER = edit_distance(reference_words, predicted_words) / len(reference_words)
```

Both use the `jiwer` library. Lower is better; 0.0 = perfect.

---

## Training Configuration (Baseline)

| Parameter        | Value          |
|------------------|----------------|
| Optimizer        | AdamW          |
| Learning rate    | 5e-4           |
| Weight decay     | 1e-4           |
| Batch size       | 8              |
| Max epochs       | 15             |
| Gradient clip    | 5.0            |
| Random seed      | 42             |
| Device           | auto (MPS/CPU) |
| num_workers      | 0              |

---

## Experiment Results

> Results will be recorded here after the full training run completes.

### Sanity Check (200 train / 50 val / 1 epoch)

| Metric     | Value |
|------------|-------|
| train_loss | TBD   |
| val_loss   | TBD   |
| val_CER    | TBD   |
| val_WER    | TBD   |

### Full Baseline Training

| Epoch | train_loss | val_loss | val_CER | val_WER |
|-------|-----------|----------|---------|---------|
| TBD   | TBD       | TBD      | TBD     | TBD     |

### Final Test Set Results

| Metric   | Value |
|----------|-------|
| Test CER | TBD   |
| Test WER | TBD   |
