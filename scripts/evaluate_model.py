#!/usr/bin/env python3
"""
evaluate_model.py
=================
Evaluate a trained checkpoint on the IAM test split.

Loads the best checkpoint, rebuilds the tokenizer from the saved vocabulary
(no data leakage), runs inference on the test split using the locally-cached
HuggingFace dataset, and reports CER/WER together with sample predictions.

The IAM-line dataset is read from the HuggingFace local cache
(~/.cache/huggingface/datasets/ by default) — no CDN requests required
after the first download.

Usage
-----
    python scripts/evaluate_model.py --checkpoint checkpoints/best.pt
    python scripts/evaluate_model.py --checkpoint checkpoints/best.pt --n-samples 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn
from src.dataset.iam_loader import load_iam_splits
from src.evaluation.metrics import evaluate
from src.inference.greedy_decoder import GreedyDecoder
from src.models.crnn import CRNN
from src.preprocessing.image_transforms import ImagePreprocessor
from src.training.checkpoint import load_checkpoint, rebuild_tokenizer_from_checkpoint
from src.training.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained HTR checkpoint on the IAM test set"
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of test samples to evaluate (default: full test split)",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-show", type=int, default=10,
                   help="Number of prediction examples to print")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="test",
                   choices=["test", "validation"],
                   help="Which split to evaluate on (default: test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["inference"].get("device", "auto"))
    cache_dir = cfg.get("dataset", {}).get("cache_dir", None)

    # ── Load checkpoint ───────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    state = load_checkpoint(args.checkpoint, device=device)
    print(f"  Saved at epoch {state['epoch']}  val_loss={state['val_loss']:.4f}")

    # ── Reconstruct tokenizer from saved vocab (no leakage) ───────────
    tokenizer = rebuild_tokenizer_from_checkpoint(state)
    print(f"  vocab_size={tokenizer.vocab_size}")

    # ── Reconstruct model ─────────────────────────────────────────────
    model = CRNN.from_config(tokenizer.vocab_size, state["model_cfg"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"  {model}")

    # ── Load split from local HF cache ────────────────────────────────
    print(f"\nLoading '{args.split}' split from HuggingFace local cache …")
    print(f"  cache_dir = {cache_dir or '~/.cache/huggingface/datasets/'}")
    splits = load_iam_splits(cache_dir=cache_dir)
    hf_split = splits[args.split]

    if args.n_samples is not None and args.n_samples < len(hf_split):
        hf_split = hf_split.select(range(args.n_samples))

    print(f"  {args.split} samples: {len(hf_split)}")

    # ── DataLoader ────────────────────────────────────────────────────
    preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"])
    ds     = IAMTorchDataset(hf_split, preprocessor, tokenizer)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=build_collate_fn(pad_value=0.0), num_workers=0,
    )

    # ── Inference ─────────────────────────────────────────────────────
    decoder   = GreedyDecoder(tokenizer)
    all_refs, all_hyps = [], []

    print(f"\nRunning inference on {len(hf_split)} samples …")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images       = batch["images"].to(device)
            image_widths = batch["image_widths"].to(device)
            logits       = model(images)
            hyps         = decoder.decode_batch(logits.cpu())
            all_refs.extend(batch["texts"])
            all_hyps.extend(hyps)
            sys.stdout.write(f"\r  batch {batch_idx+1}/{len(loader)}")
            sys.stdout.flush()
    print()

    # ── Metrics ───────────────────────────────────────────────────────
    result = evaluate(all_refs, all_hyps, include_samples=True)
    print(f"\n{'='*60}")
    print(f"  {args.split.capitalize()} Results  (n={result.n_samples})")
    print(f"{'='*60}")
    print(f"  CER : {result.cer:.4f}  ({result.cer*100:.2f}%)")
    print(f"  WER : {result.wer:.4f}  ({result.wer*100:.2f}%)")
    print(f"{'='*60}\n")

    # ── Sample predictions ────────────────────────────────────────────
    n_show = min(args.n_show, len(result.samples))
    print(f"  Sample Predictions ({n_show} examples)")
    print(f"  {'─'*56}")
    for i, s in enumerate(result.samples[:n_show]):
        print(f"\n  [{i+1}]")
        print(f"  Ground truth : {s.reference!r}")
        print(f"  Prediction   : {s.hypothesis!r}")
        print(f"  CER={s.cer:.3f}  WER={s.wer:.3f}")


if __name__ == "__main__":
    main()
