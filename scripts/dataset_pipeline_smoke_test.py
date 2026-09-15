#!/usr/bin/env python3
"""
dataset_pipeline_smoke_test.py
===============================
End-to-end smoke test for the preprocessing + tokenization + DataLoader pipeline.

This script:
  1. Fetches N rows from the IAM-line training split via the HF Datasets Viewer API
     (no full dataset download required)
  2. Builds the character tokenizer from those training texts
  3. Creates an IAMTorchDataset
  4. Creates a DataLoader with the custom collate function
  5. Fetches one batch and prints diagnostics

Does NOT train a model. Does NOT write data to the repository.

Usage
-----
    python scripts/dataset_pipeline_smoke_test.py [--n-samples N] [--batch-size B]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import requests
from PIL import Image
import torch
from torch.utils.data import DataLoader

# ── Make src importable from project root ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import ImagePreprocessor
from src.dataset.tokenizer import CharTokenizer
from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn

HF_API_BASE = "https://datasets-server.huggingface.co"
DATASET_ID = "Teklia/IAM-line"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def fetch_rows_via_api(split: str, n: int) -> list[dict]:
    """Fetch n rows from a split using the HF Datasets Viewer REST API."""
    resp = requests.get(
        f"{HF_API_BASE}/rows",
        params={
            "dataset": DATASET_ID,
            "config": "default",
            "split": split,
            "offset": 0,
            "length": n,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [r["row"] for r in resp.json().get("rows", [])]


def load_image_from_url(src_dict: dict) -> Image.Image:
    url = src_dict.get("src", "")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


class ApiFetchedSplit:
    """Minimal HF-split-compatible wrapper around rows fetched from the API."""

    def __init__(self, rows: list[dict], load_images: bool = True):
        self._rows = rows
        self._images: list[Image.Image] = []
        self._texts: list[str] = []

        print(f"  Downloading {len(rows)} images from HF CDN …")
        for i, row in enumerate(rows):
            img = load_image_from_url(row["image"])
            self._images.append(img)
            self._texts.append(row["text"])
            print(f"    [{i+1}/{len(rows)}]  {img.size[0]}×{img.size[1]}px  "
                  f"'{row['text'][:40]}{'…' if len(row['text'])>40 else ''}'")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return {"image": self._images[idx], "text": self._texts[idx]}

    @property
    def texts(self) -> list[str]:
        return self._texts


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: preprocessing + tokenizer + DataLoader pipeline"
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of training rows to fetch from HF API (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="DataLoader batch size for the smoke test (default: 4)"
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Dataset Pipeline Smoke Test")
    print("  Source : Teklia/IAM-line  (HF Datasets Viewer API)")
    print("=" * 64)

    # ── Step 1: Fetch training rows from HF API ───────────────────────────────
    print(f"\n[1/5] Fetching {args.n_samples} training rows from HF API …")
    rows = fetch_rows_via_api("train", args.n_samples)
    split = ApiFetchedSplit(rows)
    train_texts = split.texts
    print(f"  ✓  Fetched {len(split)} samples")

    # ── Step 2: Build tokenizer from training texts only ─────────────────────
    print("\n[2/5] Building character tokenizer from training texts …")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(train_texts)
    print(f"  ✓  Vocabulary size : {tokenizer.vocab_size}")
    print(f"     Blank index     : {tokenizer.blank_index}")
    print(f"     UNK index       : {tokenizer.unk_index}")
    sorted_chars = sorted(
        c for c in tokenizer.char2idx
        if c not in (tokenizer.blank_token, tokenizer.unknown_token)
    )
    print(f"     Characters      : {repr(''.join(sorted_chars))}")

    # ── Step 3: Create preprocessor and dataset ───────────────────────────────
    print("\n[3/5] Creating ImagePreprocessor and IAMTorchDataset …")
    preprocessor = ImagePreprocessor(target_height=128, mean=0.5, std=0.5)
    dataset = IAMTorchDataset(split, preprocessor, tokenizer)
    print(f"  ✓  Dataset length  : {len(dataset)}")
    print(f"     Preprocessor    : {preprocessor}")

    # ── Step 4: Create DataLoader ─────────────────────────────────────────────
    print(f"\n[4/5] Creating DataLoader (batch_size={args.batch_size}) …")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=build_collate_fn(pad_value=0.0),
        shuffle=False,
    )
    print(f"  ✓  DataLoader ready  ({len(loader)} batches)")

    # ── Step 5: Fetch and inspect first batch ─────────────────────────────────
    print("\n[5/5] Fetching and inspecting first batch …")
    batch = next(iter(loader))

    print("\n" + "─" * 64)
    print("  Batch diagnostics")
    print("─" * 64)
    print(f"  images.shape        : {tuple(batch['images'].shape)}")
    print(f"  images.dtype        : {batch['images'].dtype}")
    print(f"  individual widths   : {batch['image_widths'].tolist()}")
    print(f"  targets.shape       : {tuple(batch['targets'].shape)}")
    print(f"  target_lengths      : {batch['target_lengths'].tolist()}")
    print(f"  vocabulary size     : {tokenizer.vocab_size}")

    # Show details for the first sample in the batch
    print("\n  First sample in batch:")
    first_text = batch["texts"][0]
    first_encoded = tokenizer.encode(first_text)
    first_decoded = tokenizer.decode(first_encoded)
    print(f"    raw text          : {repr(first_text)}")
    print(f"    encoded           : {first_encoded}")
    print(f"    decoded           : {repr(first_decoded)}")
    print(f"    encode→decode OK  : {first_text == first_decoded}")

    print("\n" + "=" * 64)
    print("  Smoke test PASSED ✓")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
