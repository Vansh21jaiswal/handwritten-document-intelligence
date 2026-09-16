"""
trainer.py
==========
Training and validation loop for the CNN-BiLSTM-CTC HTR model.

Data access
-----------
Uses load_iam_splits() from src.dataset.iam_loader, which calls
datasets.load_dataset("Teklia/IAM-line"). On the first run this downloads
the full dataset as Parquet files (images embedded) to the HuggingFace
local cache (~/.cache/huggingface/datasets/ by default). All subsequent
runs read from this local cache — no network access required.

This module provides:
  - build_data_pipeline()  — tokenizer + datasets + DataLoaders
  - train_one_epoch()      — one pass over the training DataLoader
  - validate_epoch()       — loss + greedy-decoded CER/WER
  - run_training()         — full orchestrator (smoke + full modes)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn
from src.dataset.iam_loader import load_iam_splits
from src.dataset.tokenizer import CharTokenizer
from src.evaluation.metrics import compute_cer, compute_wer
from src.inference.greedy_decoder import GreedyDecoder
from src.models.crnn import CRNN
from src.preprocessing.image_transforms import ImagePreprocessor
from src.training.checkpoint import save_checkpoint
from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
from src.training.device import get_device
from src.training.seed import set_seed


# ────────────────────────────────────────────────────────────────────────────
# Data pipeline builder
# ────────────────────────────────────────────────────────────────────────────

def build_data_pipeline(
    train_hf,
    val_hf,
    cfg: dict,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[CharTokenizer, DataLoader, DataLoader]:
    """Build tokenizer + DataLoaders from HuggingFace Dataset splits.

    Tokenizer vocabulary is built from train_hf["text"] ONLY — no leakage.

    Parameters
    ----------
    train_hf : datasets.Dataset
        Training split (or subset) from load_iam_splits().
    val_hf : datasets.Dataset
        Validation split (or subset).
    cfg : dict
        Full config dict (uses cfg["tokenizer"] and cfg["preprocessing"]).
    batch_size : int
    num_workers : int

    Returns
    -------
    tokenizer, train_loader, val_loader
    """
    # Build tokenizer from training texts only
    tokenizer = CharTokenizer.from_config(cfg["tokenizer"])
    tokenizer.build_vocab(train_hf["text"])   # HF column access → List[str]
    print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}  blank_idx={tokenizer.blank_index}")

    preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"])
    collate = build_collate_fn(pad_value=0.0)

    train_ds = IAMTorchDataset(train_hf, preprocessor, tokenizer)
    val_ds   = IAMTorchDataset(val_hf,   preprocessor, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=False,
    )
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}")
    return tokenizer, train_loader, val_loader


# ────────────────────────────────────────────────────────────────────────────
# Per-epoch training / validation
# ────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """One full pass over the training DataLoader.

    Returns
    -------
    float
        Mean training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images        = batch["images"].to(device)
        targets       = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        image_widths  = batch["image_widths"].to(device)

        input_lengths = model.compute_input_lengths(image_widths)
        logits        = model(images)
        loss          = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        sys.stdout.write(
            f"\r  batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}"
        )
        sys.stdout.flush()

    print()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    decoder: GreedyDecoder,
    device: torch.device,
) -> Tuple[float, float, float]:
    """One full pass over the validation DataLoader.

    Returns
    -------
    (val_loss, val_cer, val_wer)
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_refs:  List[str] = []
    all_hyps:  List[str] = []

    for batch in loader:
        images         = batch["images"].to(device)
        targets        = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        image_widths   = batch["image_widths"].to(device)
        texts          = batch["texts"]

        input_lengths = model.compute_input_lengths(image_widths)
        logits        = model(images)

        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        n_batches  += 1

        # Decode on CPU (safe for MPS/CUDA)
        hyps = decoder.decode_batch(logits.cpu())
        all_refs.extend(texts)
        all_hyps.extend(hyps)

    val_loss = total_loss / max(n_batches, 1)
    val_cer  = compute_cer(all_refs, all_hyps)
    val_wer  = compute_wer(all_refs, all_hyps)
    return val_loss, val_cer, val_wer


# ────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ────────────────────────────────────────────────────────────────────────────

def run_training(
    cfg: dict,
    *,
    smoke: bool = False,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Set up and run the full training pipeline using locally-cached HF data.

    Parameters
    ----------
    cfg : dict
        Parsed default.yaml.
    smoke : bool
        If True, use smoke_* config values for a quick sanity check.
    max_train_samples, max_val_samples, max_epochs, batch_size : optional
        Command-line overrides; take priority over cfg values.
    """
    tcfg = cfg["training"]
    dcfg = cfg.get("dataset", {})
    device = get_device(tcfg.get("device", "auto"))

    # ── Resolve run parameters ──────────────────────────────────────
    if smoke:
        n_train  = max_train_samples or tcfg["smoke_train_samples"]
        n_val    = max_val_samples   or tcfg["smoke_val_samples"]
        n_epochs = max_epochs        or tcfg["smoke_epochs"]
    else:
        n_train  = max_train_samples or tcfg.get("max_train_samples") or None
        n_val    = max_val_samples   or tcfg.get("max_val_samples")   or None
        n_epochs = max_epochs        or tcfg["max_epochs"]

    bs          = batch_size or tcfg["batch_size"]
    grad_clip   = tcfg.get("grad_clip", 5.0)
    num_workers = tcfg.get("num_workers", 0)
    seed        = tcfg.get("random_seed", 42)
    ckpt_dir    = tcfg.get("checkpoint_dir", "checkpoints")
    cache_dir   = dcfg.get("cache_dir", None)   # None → ~/.cache/huggingface/

    set_seed(seed)

    print(f"\n{'='*62}")
    mode = "SMOKE" if smoke else "FULL"
    print(f"  HTR Training — {mode} mode")
    print(f"  train≤{n_train or 'all'}  val≤{n_val or 'all'}  epochs={n_epochs}  batch={bs}")
    print(f"{'='*62}\n")

    # ── Load dataset from local HF cache ─────────────────────────────
    print("[1/5] Loading IAM-line splits from HuggingFace cache …")
    print(f"  cache_dir = {cache_dir or '~/.cache/huggingface/datasets/'}")
    splits = load_iam_splits(cache_dir=cache_dir)

    train_hf = splits["train"]
    val_hf   = splits["validation"]

    # Optionally cap split sizes (smoke mode or CLI override)
    if n_train is not None and n_train < len(train_hf):
        train_hf = train_hf.select(range(n_train))
    if n_val is not None and n_val < len(val_hf):
        val_hf = val_hf.select(range(n_val))

    print(f"  train={len(train_hf)}  val={len(val_hf)}")

    # ── Build tokenizer + DataLoaders ─────────────────────────────────
    print("\n[2/5] Building tokenizer and DataLoaders …")
    tokenizer, train_loader, val_loader = build_data_pipeline(
        train_hf, val_hf, cfg, bs, num_workers
    )

    # ── Model ──────────────────────────────────────────────────────────
    print("\n[3/5] Creating model …")
    model = CRNN.from_config(tokenizer.vocab_size, cfg["model"]).to(device)
    print(f"  {model}")

    # ── Optimizer + Loss + Decoder ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    criterion = build_ctc_loss(blank=tokenizer.blank_index)
    decoder   = GreedyDecoder(tokenizer)

    best_val_loss = float("inf")
    t_total_start = time.time()

    print(f"\n[4/5] Training for {n_epochs} epoch(s) …")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{n_epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        val_loss, val_cer, val_wer = validate_epoch(
            model, val_loader, criterion, decoder, device
        )
        elapsed = time.time() - t0

        print(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_CER={val_cer:.4f}  "
            f"val_WER={val_wer:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # Checkpoint every epoch
        save_checkpoint(
            ckpt_dir, "latest.pt",
            model, optimizer, epoch, val_loss, tokenizer, cfg["model"]
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir, "best.pt",
                model, optimizer, epoch, val_loss, tokenizer, cfg["model"]
            )
            print(f"  ★ New best val_loss={val_loss:.4f} → saved best.pt")

    total_time = time.time() - t_total_start
    print(f"\n[5/5] Training complete.")
    print(f"  Best val_loss  = {best_val_loss:.4f}")
    print(f"  Total time     = {total_time/60:.1f} min")
    print(f"  Checkpoints in : {Path(ckpt_dir).resolve()}")
    print(f"\n{'='*62}\n")
