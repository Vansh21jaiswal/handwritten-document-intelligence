#!/usr/bin/env python3
"""
visualize_samples.py
====================
Displays a grid of handwritten line images alongside their ground-truth
transcriptions from the Teklia/IAM-line dataset.

Uses the HuggingFace Datasets Viewer REST API — no full dataset download needed.

Usage
-----
    python scripts/visualize_samples.py [--split SPLIT] [--n N] [--save PATH]

Arguments
---------
--split   : One of 'train', 'validation', 'test' (default: train)
--n       : Number of samples to display (default: 6)
--save    : If provided, save the figure to this path instead of showing it.
            Example: --save outputs/sample_grid.png

No dataset files are written to the repository.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
from PIL import Image

# ── Make src importable when running from the project root ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HF_API_BASE = "https://datasets-server.huggingface.co"
DATASET_ID = "Teklia/IAM-line"


def _fetch_rows(split: str, n: int) -> list[dict]:
    resp = requests.get(
        f"{HF_API_BASE}/rows",
        params={"dataset": DATASET_ID, "config": "default", "split": split,
                "offset": 0, "length": n},
        timeout=30,
    )
    resp.raise_for_status()
    return [r["row"] for r in resp.json().get("rows", [])]


def _load_image(src_dict: dict) -> Image.Image:
    url = src_dict.get("src", "")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def visualize(split_name: str, n: int, save_path: str | None) -> None:
    print(f"Fetching {n} rows from '{split_name}' split via HF API …")
    rows = _fetch_rows(split_name, n)

    fig = plt.figure(figsize=(14, 2.2 * len(rows)))
    fig.suptitle(
        f"IAM-line — '{split_name}' split  (source: Teklia/IAM-line on HF Hub)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(len(rows), 1, hspace=0.65)

    for i, row in enumerate(rows):
        ax = fig.add_subplot(gs[i])
        img = _load_image(row["image"])
        text = row["text"]

        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            repr(text),
            fontsize=8, labelpad=4, fontstyle="italic",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.7)

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Figure saved → {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize IAM-line samples via HF Datasets Viewer API."
    )
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=6,
                        help="Number of samples (default: 6)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure here instead of displaying")
    args = parser.parse_args()
    visualize(args.split, args.n, args.save)


if __name__ == "__main__":
    main()
