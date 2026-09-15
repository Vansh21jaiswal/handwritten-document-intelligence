#!/usr/bin/env python3
"""
inspect_dataset.py
==================
Dataset inspection script for Teklia/IAM-line.

Uses the HuggingFace Datasets Viewer REST API so that:
  - No full dataset download is needed
  - No large parquet index shards are fetched
  - A handful of rows are returned as JSON in seconds

Reports:
  - Number of samples in each split
  - Image dimensions for several samples
  - Text/label examples
  - Min/max/average transcription length (over fetched samples)
  - Character vocabulary (over fetched samples)
  - Number of unique characters

Usage
-----
    python scripts/inspect_dataset.py [--samples N]

Requires network access to https://datasets-server.huggingface.co
No data is written into the repository.
"""

from __future__ import annotations

import argparse
import base64
import io
import statistics
import sys
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

# ── Make src importable when running from the project root ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── HF Datasets Viewer API constants ─────────────────────────────────────────
HF_API_BASE = "https://datasets-server.huggingface.co"
DATASET_ID = "Teklia/IAM-line"

# Authoritative Aachen split sizes
KNOWN_SPLIT_SIZES = {
    "train": 6482,
    "validation": 976,
    "test": 2915,
}
SPLIT_NAMES = ["train", "validation", "test"]


# ────────────────────────────────────────────────────────────────────────────
# API helpers
# ────────────────────────────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict) -> dict:
    url = f"{HF_API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_rows(split: str, n: int) -> list[dict]:
    """Return up to n row dicts from the HF Datasets Viewer API."""
    data = _api_get("rows", {
        "dataset": DATASET_ID,
        "config": "default",
        "split": split,
        "offset": 0,
        "length": n,
    })
    return [row["row"] for row in data.get("rows", [])]


def _decode_image(src: str | dict) -> Image.Image | None:
    """Fetch and decode an image from the HF cached-assets CDN URL."""
    try:
        if isinstance(src, dict):
            src = src.get("src", "")
        if not src:
            return None
        # HF returns signed CDN URLs for cached image assets
        if src.startswith("http"):
            resp = requests.get(src, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        # Fallback: base64 data URI
        if src.startswith("data:"):
            _, b64 = src.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────────────────────
# Display helpers
# ────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _inspect_split(split_name: str, rows: list[dict]) -> list[str]:
    texts = [r.get("text", "") for r in rows]

    # ── Image dimensions ─────────────────────────────────────────────────────
    print(f"\n  Image dimensions ({len(rows)} samples from API):")
    widths, heights = [], []
    for i, row in enumerate(rows):
        img_data = row.get("image")
        img = _decode_image(img_data) if img_data else None
        if img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
            print(f"    [{i:>3}]  {w:>5} × {h:<4}px")
        else:
            print(f"    [{i:>3}]  (image decode skipped — see note)")

    if len(widths) > 1:
        print(f"\n  Width  — min: {min(widths)}  max: {max(widths)}  "
              f"mean: {statistics.mean(widths):.1f}")
        print(f"  Height — min: {min(heights)}  max: {max(heights)}  "
              f"mean: {statistics.mean(heights):.1f}")

    # ── Label examples ───────────────────────────────────────────────────────
    print(f"\n  Transcription examples:")
    for i, t in enumerate(texts):
        snippet = t[:80] + ("…" if len(t) > 80 else "")
        print(f"    [{i:>3}]  {repr(snippet)}")

    # ── Length statistics ────────────────────────────────────────────────────
    lengths = [len(t) for t in texts if t]
    if lengths:
        print(f"\n  Transcription length (chars, {len(lengths)} samples):")
        print(f"    min  : {min(lengths)}")
        print(f"    max  : {max(lengths)}")
        print(f"    mean : {statistics.mean(lengths):.2f}")
        if len(lengths) > 1:
            print(f"    stdev: {statistics.stdev(lengths):.2f}")

    return texts


def _global_vocab(all_texts: list[str]) -> None:
    _section("Character Vocabulary (sampled from all splits via API)")

    char_counter: Counter = Counter()
    for text in all_texts:
        char_counter.update(text)

    vocab = sorted(char_counter.keys())
    print(f"\n  Unique characters : {len(vocab)}")
    print(f"  Total characters  : {sum(char_counter.values()):,}")

    print(f"\n  Vocabulary (sorted):")
    row_size = 16
    for i in range(0, len(vocab), row_size):
        chunk = vocab[i : i + row_size]
        print("    " + "  ".join(f"{repr(c):<5}" for c in chunk))

    print(f"\n  Top-10 most frequent characters:")
    for ch, count in char_counter.most_common(10):
        print(f"    {repr(ch):<8} {count:>8,}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Teklia/IAM-line via HF Datasets Viewer API (no download)."
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Rows to fetch per split from the API (default: 5)"
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Handwritten Document Intelligence — Dataset Inspection")
    print("  Dataset : Teklia/IAM-line  (HF Datasets Viewer API)")
    print("=" * 64)

    # ── Split overview ────────────────────────────────────────────────────────
    _section("Split Overview (authoritative Aachen counts)")
    total = sum(KNOWN_SPLIT_SIZES.values())
    for name in SPLIT_NAMES:
        print(f"  {name:<12} {KNOWN_SPLIT_SIZES[name]:>6,} samples")
    print(f"  {'TOTAL':<12} {total:>6,} samples")

    # ── Per-split inspection ──────────────────────────────────────────────────
    all_texts: list[str] = []
    for name in SPLIT_NAMES:
        _section(f"Split: {name}")
        print(f"  Fetching {args.samples} rows from HF API …")
        try:
            rows = _fetch_rows(name, args.samples)
            texts = _inspect_split(name, rows)
            all_texts.extend(texts)
        except requests.HTTPError as e:
            print(f"  ⚠  API error: {e}")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    if all_texts:
        _global_vocab(all_texts)

    print("\n" + "=" * 64)
    print("  Inspection complete.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
