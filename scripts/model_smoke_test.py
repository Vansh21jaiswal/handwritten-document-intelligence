#!/usr/bin/env python3
"""
model_smoke_test.py
===================
End-to-end model smoke test:

  image → CNN → BiLSTM → logits → CTC loss → backpropagation

Steps
-----
1. Load a small subset of IAM training data via HF Datasets Viewer API.
2. Build the character tokenizer from training texts only.
3. Create ImagePreprocessor, IAMTorchDataset, and DataLoader.
4. Instantiate the CRNN model.
5. Run one forward pass and compute CTC loss.
6. Run one optimizer step (backprop).
7. Print diagnostics.
8. Run a 1-epoch smoke training experiment.

No model checkpoints are saved.
No dataset files are stored in the repository.

Usage
-----
    python scripts/model_smoke_test.py [--n-samples N] [--batch-size B]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import requests
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn
from src.dataset.tokenizer import CharTokenizer
from src.models.crnn import CRNN
from src.preprocessing.image_transforms import ImagePreprocessor
from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
from src.training.device import get_device
from src.training.trainer import run_training

HF_API_BASE = "https://datasets-server.huggingface.co"
DATASET_ID = "Teklia/IAM-line"


# ────────────────────────────────────────────────────────────────────────────
# API helpers
# ────────────────────────────────────────────────────────────────────────────

def fetch_rows(split: str, n: int) -> list:
    resp = requests.get(
        f"{HF_API_BASE}/rows",
        params={"dataset": DATASET_ID, "config": "default",
                "split": split, "offset": 0, "length": n},
        timeout=30,
    )
    resp.raise_for_status()
    return [r["row"] for r in resp.json().get("rows", [])]


def load_pil(src_dict: dict) -> Image.Image:
    url = src_dict.get("src", "")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


class ApiFetchedSplit:
    def __init__(self, rows: list) -> None:
        self._images, self._texts = [], []
        print(f"  Downloading {len(rows)} images …")
        for i, row in enumerate(rows):
            self._images.append(load_pil(row["image"]))
            self._texts.append(row["text"])
            sys.stdout.write(f"\r  [{i+1}/{len(rows)}]")
            sys.stdout.flush()
        print()

    def __len__(self): return len(self._texts)
    def __getitem__(self, idx): return {"image": self._images[idx], "text": self._texts[idx]}
    @property
    def texts(self): return self._texts


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRNN model smoke test — end-to-end pipeline verification"
    )
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Training rows to fetch for the forward-pass test (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for the forward-pass test (default: 4)")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 64)
    print("  HTR Model Smoke Test — CNN-BiLSTM-CTC")
    print("=" * 64)

    # ── Device ───────────────────────────────────────────────────────
    device = get_device(cfg["training"].get("device", "auto"))

    # ── Step 1: Data ─────────────────────────────────────────────────
    print(f"\n[1/6] Fetching {args.n_samples} training rows from HF API …")
    rows = fetch_rows("train", args.n_samples)
    split = ApiFetchedSplit(rows)

    # ── Step 2: Tokenizer (train texts only) ─────────────────────────
    print("\n[2/6] Building tokenizer from training texts …")
    tokenizer = CharTokenizer.from_config(cfg["tokenizer"])
    tokenizer.build_vocab(split.texts)
    print(f"  vocab_size = {tokenizer.vocab_size}")
    print(f"  blank_index = {tokenizer.blank_index}")

    # ── Step 3: Dataset + DataLoader ─────────────────────────────────
    print("\n[3/6] Creating dataset and DataLoader …")
    preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"])
    dataset = IAMTorchDataset(split, preprocessor, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=build_collate_fn(pad_value=0.0),
        shuffle=False,
    )
    batch = next(iter(loader))
    print(f"  batch images shape : {tuple(batch['images'].shape)}")
    print(f"  individual widths  : {batch['image_widths'].tolist()}")
    print(f"  targets shape      : {tuple(batch['targets'].shape)}")
    print(f"  target_lengths     : {batch['target_lengths'].tolist()}")

    # ── Step 4: Model ─────────────────────────────────────────────────
    print("\n[4/6] Instantiating CRNN model …")
    model = CRNN.from_config(tokenizer.vocab_size, cfg["model"]).to(device)
    print(f"  {model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total parameters   : {total_params:,}")
    print(f"  trainable params   : {trainable:,}")

    # ── Step 5: Forward pass ──────────────────────────────────────────
    print("\n[5/6] Forward pass + CTC loss …")
    images = batch["images"].to(device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    image_widths = batch["image_widths"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images)                                     # (T, B, V)
        input_lengths = model.compute_input_lengths(image_widths)  # (B,)

    criterion = build_ctc_loss(blank=tokenizer.blank_index)
    loss_before = compute_ctc_loss(
        criterion, logits, targets, input_lengths, target_lengths
    )

    print(f"\n  ─── Forward pass results ───")
    print(f"  device             : {device}")
    print(f"  input shape        : {tuple(images.shape)}")
    print(f"  logits shape       : {tuple(logits.shape)}")
    print(f"  input_lengths      : {input_lengths.tolist()}")
    print(f"  target_lengths     : {target_lengths.tolist()}")
    print(f"  vocabulary size    : {tokenizer.vocab_size}")
    print(f"  CTC loss (before)  : {loss_before.item():.4f}")

    # ── Step 6: Optimizer step ────────────────────────────────────────
    print("\n[6/6] Optimizer step (backpropagation) …")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    optimizer.zero_grad()
    logits_train = model(images)
    loss_train = compute_ctc_loss(
        criterion, logits_train, targets, input_lengths, target_lengths
    )
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
    optimizer.step()

    # Measure loss after one step
    model.eval()
    with torch.no_grad():
        logits_after = model(images)
        loss_after = compute_ctc_loss(
            criterion, logits_after, targets, input_lengths, target_lengths
        )

    print(f"  CTC loss (before)  : {loss_before.item():.4f}")
    print(f"  CTC loss (after 1 step) : {loss_after.item():.4f}")
    print(f"  Backpropagation    : ✓")

    print("\n" + "=" * 64)
    print("  Forward-pass smoke test PASSED ✓")
    print("=" * 64)

    # ── Smoke training experiment ─────────────────────────────────────
    print("\n\n" + "=" * 64)
    print("  Starting 1-epoch smoke training experiment …")
    print("=" * 64)
    run_training(cfg, smoke=True)


if __name__ == "__main__":
    main()
