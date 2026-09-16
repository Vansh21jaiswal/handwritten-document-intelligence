#!/usr/bin/env python3
"""
train_baseline.py
=================
Main training script for the CNN-BiLSTM-CTC HTR baseline.

Downloads the IAM line dataset from the HuggingFace Datasets Viewer API,
builds the tokenizer from training texts only, trains the model, saves
checkpoints, and reports final validation metrics.

Usage
-----
    python scripts/train_baseline.py                         # full training
    python scripts/train_baseline.py --max-train 500 --max-val 100 --epochs 1
    python scripts/train_baseline.py --batch-size 4 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN-BiLSTM-CTC HTR model on IAM lines."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max_epochs from config",
    )
    parser.add_argument(
        "--max-train", type=int, default=None,
        dest="max_train_samples",
        help="Cap number of training samples (default: full split = 6482)",
    )
    parser.add_argument(
        "--max-val", type=int, default=None,
        dest="max_val_samples",
        help="Cap number of validation samples (default: full split = 976)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        dest="batch_size",
        help="Override batch_size from config",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick sanity check: 200 train, 50 val, 1 epoch",
    )
    return parser.parse_args()


def print_config(cfg: dict, args) -> None:
    """Print the fully-resolved training configuration before the run starts."""
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    n_train = (args.max_train_samples or tcfg.get("max_train_samples") or 6482) if not args.smoke else (args.max_train_samples or tcfg["smoke_train_samples"])
    n_val   = (args.max_val_samples   or tcfg.get("max_val_samples")   or 976)  if not args.smoke else (args.max_val_samples   or tcfg["smoke_val_samples"])
    n_epochs = (args.epochs or tcfg["max_epochs"]) if not args.smoke else (args.epochs or tcfg["smoke_epochs"])
    bs = args.batch_size or tcfg["batch_size"]

    import torch
    if tcfg.get("device", "auto") == "auto":
        if torch.cuda.is_available():       dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): dev = "mps"
        else:                               dev = "cpu"
    else:
        dev = tcfg["device"]

    print("\n" + "="*62)
    print("  BASELINE TRAINING — RESOLVED CONFIGURATION")
    print("="*62)
    print(f"  device              : {dev}")
    print(f"  train samples       : {n_train}")
    print(f"  validation samples  : {n_val}")
    print(f"  epochs              : {n_epochs}")
    print(f"  batch size          : {bs}")
    print(f"  optimizer           : AdamW")
    print(f"  learning rate       : {tcfg['learning_rate']}")
    print(f"  weight decay        : {tcfg['weight_decay']}")
    print(f"  gradient clip       : {tcfg.get('grad_clip', 5.0)}")
    print(f"  random seed         : {tcfg.get('random_seed', 42)}")
    print(f"  num workers         : {tcfg.get('num_workers', 0)}")
    print(f"  checkpoint dir      : {tcfg.get('checkpoint_dir', 'checkpoints')}")
    print(f"  model               : CNN-BiLSTM-CTC  (CRNN)")
    print(f"  cnn channels        : {mcfg['cnn_channels']}")
    print(f"  rnn hidden          : {mcfg['rnn_hidden']}  (× 2 bidirectional)")
    print(f"  rnn layers          : {mcfg['rnn_layers']}")
    print("="*62 + "\n")


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print_config(cfg, args)

    run_training(
        cfg,
        smoke=args.smoke,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
