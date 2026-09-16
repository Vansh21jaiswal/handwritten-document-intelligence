#!/usr/bin/env python3
"""
show_predictions.py
===================
Load a trained checkpoint and display validation/test examples with predictions.

Reads from the locally-cached HuggingFace dataset — no CDN requests after
the first download.

For each example shows:
  - handwritten image
  - ground truth transcription
  - model prediction
  - per-sample CER

Saves visualization to outputs/predictions_<timestamp>.png

Usage
-----
    python scripts/show_predictions.py --checkpoint checkpoints/best.pt
    python scripts/show_predictions.py --checkpoint checkpoints/best.pt --n-samples 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn
from src.dataset.iam_loader import load_iam_splits
from src.evaluation.metrics import compute_cer
from src.inference.greedy_decoder import GreedyDecoder
from src.models.crnn import CRNN
from src.preprocessing.image_transforms import ImagePreprocessor
from src.training.checkpoint import load_checkpoint, rebuild_tokenizer_from_checkpoint
from src.training.device import get_device


def parse_args():
    p = argparse.ArgumentParser(
        description="Show validation/test predictions from a trained checkpoint"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="validation",
                   choices=["validation", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg.get("dataset", {}).get("cache_dir", None)
    device    = get_device("cpu")   # CPU is sufficient for inference display

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    state     = load_checkpoint(args.checkpoint, device=device)
    tokenizer = rebuild_tokenizer_from_checkpoint(state)
    model     = CRNN.from_config(tokenizer.vocab_size, state["model_cfg"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"  epoch={state['epoch']}  val_loss={state['val_loss']:.4f}  vocab={tokenizer.vocab_size}")

    # Load split from local HF cache
    print(f"\nLoading '{args.split}' from HuggingFace local cache …")
    splits   = load_iam_splits(cache_dir=cache_dir)
    hf_split = splits[args.split].select(range(args.n_samples))
    print(f"  Selected {len(hf_split)} samples")

    # Build dataset and dataloader
    preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"])
    ds     = IAMTorchDataset(hf_split, preprocessor, tokenizer)
    loader = DataLoader(
        ds, batch_size=4, shuffle=False,
        collate_fn=build_collate_fn(), num_workers=0,
    )

    # Run inference
    decoder   = GreedyDecoder(tokenizer)
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            hyps   = decoder.decode_batch(logits.cpu())
            all_refs.extend(batch["texts"])
            all_hyps.extend(hyps)

    # Build visualisation
    n_show = min(args.n_samples, len(all_refs))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.8 * n_show))
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Fetch original PIL image from HF split for display
        raw_img = hf_split[i]["image"].convert("L")
        cer_i   = compute_cer([all_refs[i]], [all_hyps[i]])

        ax.imshow(raw_img, cmap="gray", aspect="auto")
        ax.axis("off")
        ax.set_title(
            f"GT:   {all_refs[i]!r}\n"
            f"PRED: {all_hyps[i]!r}   CER={cer_i:.3f}",
            loc="left", fontsize=9, pad=4,
        )

    plt.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"predictions_{ts}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nVisualization saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
