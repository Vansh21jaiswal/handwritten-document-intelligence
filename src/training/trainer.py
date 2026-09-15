"""
trainer.py
==========
Minimal training and validation loop for the CNN-BiLSTM-CTC HTR model.

This module provides:
  - train_one_epoch()  — one pass over the training DataLoader
  - validate()         — one pass over the validation DataLoader
  - run_training()     — orchestrates the full training setup

The model is NOT saved to disk in this file. Checkpointing will be
added in a later sprint.

Usage
-----
>>> from src.training.trainer import run_training
>>> run_training(cfg, smoke=True)  # runs a 1-epoch smoke experiment
"""

from __future__ import annotations

import io
import sys
import time
from typing import Dict

import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset

from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn
from src.dataset.tokenizer import CharTokenizer
from src.models.crnn import CRNN
from src.preprocessing.image_transforms import ImagePreprocessor
from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
from src.training.device import get_device

# ─── HF Datasets Viewer API (no full dataset download) ───────────────────────
HF_API_BASE = "https://datasets-server.huggingface.co"
DATASET_ID = "Teklia/IAM-line"


# ────────────────────────────────────────────────────────────────────────────
# Data fetching helpers (API-based, no disk storage)
# ────────────────────────────────────────────────────────────────────────────

def _fetch_rows(split: str, n: int) -> list:
    """Fetch rows from HF API, chunking to obey the 100-row limit per request."""
    rows = []
    chunk_size = 100
    for offset in range(0, n, chunk_size):
        length = min(chunk_size, n - offset)
        resp = requests.get(
            f"{HF_API_BASE}/rows",
            params={"dataset": DATASET_ID, "config": "default",
                    "split": split, "offset": offset, "length": length},
            timeout=30,
        )
        resp.raise_for_status()
        rows.extend(r["row"] for r in resp.json().get("rows", []))
    return rows


def _load_pil(src_dict: dict) -> Image.Image:
    url = src_dict.get("src", "")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


class _ApiFetchedSplit:
    """Minimal HF-split-compatible wrapper around rows fetched from the API."""

    def __init__(self, rows: list, verbose: bool = True) -> None:
        self._images: list = []
        self._texts: list = []
        for i, row in enumerate(rows):
            img = _load_pil(row["image"])
            self._images.append(img)
            self._texts.append(row["text"])
            if verbose:
                sys.stdout.write(f"\r  Downloaded {i+1}/{len(rows)} images")
                sys.stdout.flush()
        if verbose:
            print()

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> Dict:
        return {"image": self._images[idx], "text": self._texts[idx]}

    @property
    def texts(self) -> list:
        return self._texts


# ────────────────────────────────────────────────────────────────────────────
# Training loop helpers
# ────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one full pass over the training DataLoader.

    Parameters
    ----------
    model       : CRNN model in training mode
    loader      : DataLoader with the custom collate function
    criterion   : nn.CTCLoss instance
    optimizer   : torch optimizer
    device      : target device
    grad_clip   : max L2 norm for gradient clipping (0 = no clipping)

    Returns
    -------
    float
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)          # (B, 1, H, W_max)
        targets = batch["targets"].to(device)        # (sum_T,)
        target_lengths = batch["target_lengths"].to(device)  # (B,)
        image_widths = batch["image_widths"].to(device)      # (B,)

        # Compute CTC input lengths from image widths
        input_lengths = model.compute_input_lengths(image_widths)  # (B,)

        # Forward pass
        logits = model(images)                       # (T, B, vocab_size)

        # CTC loss
        loss = compute_ctc_loss(
            criterion, logits, targets, input_lengths, target_lengths
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        sys.stdout.write(
            f"\r  batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}"
        )
        sys.stdout.flush()

    print()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> float:
    """Evaluate the model on a validation DataLoader.

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        image_widths = batch["image_widths"].to(device)

        input_lengths = model.compute_input_lengths(image_widths)
        logits = model(images)
        loss = compute_ctc_loss(
            criterion, logits, targets, input_lengths, target_lengths
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ────────────────────────────────────────────────────────────────────────────
# Top-level training orchestrator
# ────────────────────────────────────────────────────────────────────────────

def run_training(cfg: dict, smoke: bool = False) -> None:
    """Set up and run the full training pipeline.

    Parameters
    ----------
    cfg : dict
        Parsed default.yaml (full config dict).
    smoke : bool
        If True, uses smoke_train_samples / smoke_val_samples / smoke_epochs
        from cfg["training"] for a quick end-to-end verification run.
    """
    tcfg = cfg["training"]
    device = get_device(tcfg.get("device", "auto"))

    n_train = tcfg["smoke_train_samples"] if smoke else None
    n_val = tcfg["smoke_val_samples"] if smoke else None
    n_epochs = tcfg["smoke_epochs"] if smoke else tcfg["max_epochs"]
    batch_size = tcfg["batch_size"]

    print(f"\n{'='*60}")
    mode = "SMOKE" if smoke else "FULL"
    print(f"  HTR Training — {mode} mode")
    print(f"  train_samples={n_train}  val_samples={n_val}  epochs={n_epochs}")
    print(f"{'='*60}\n")

    # ── Fetch data via HF API ─────────────────────────────────────────
    print(f"[1/5] Fetching {n_train} training rows …")
    train_rows = _fetch_rows("train", n_train or 6482)
    train_split = _ApiFetchedSplit(train_rows)

    print(f"\n[1/5] Fetching {n_val} validation rows …")
    val_rows = _fetch_rows("validation", n_val or 976)
    val_split = _ApiFetchedSplit(val_rows)

    # ── Tokenizer (train texts only) ──────────────────────────────────
    print("\n[2/5] Building tokenizer from training texts …")
    tokenizer = CharTokenizer.from_config(cfg["tokenizer"])
    tokenizer.build_vocab(train_split.texts)
    print(f"  vocab_size = {tokenizer.vocab_size}")

    # ── Preprocessing + Dataset + DataLoader ──────────────────────────
    print("\n[3/5] Creating datasets and dataloaders …")
    preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"])
    train_ds = IAMTorchDataset(train_split, preprocessor, tokenizer)
    val_ds = IAMTorchDataset(val_split, preprocessor, tokenizer)

    collate = build_collate_fn(pad_value=0.0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    print("\n[4/5] Creating model …")
    model = CRNN.from_config(tokenizer.vocab_size, cfg["model"]).to(device)
    print(f"  {model}")

    # ── Optimizer + Loss ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = build_ctc_loss(blank=tokenizer.blank_index)
    grad_clip = tcfg.get("grad_clip", 5.0)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n[5/5] Training for {n_epochs} epoch(s) …")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{n_epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

    print(f"\n{'='*60}")
    print("  Training complete.")
    print(f"{'='*60}\n")
