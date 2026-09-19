#!/usr/bin/env python3
"""
error_analysis.py
=================
Deep dive error analysis on the test set predictions.

Computes:
  - Aggregate character-level stats (insertions, deletions, substitutions).
  - Performance broken down by string characteristics.
  - Generates markdown report.
  - Plots representative successes and failures.

Usage
-----
    python scripts/error_analysis.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
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


def plot_examples(samples, hf_split, title, out_path):
    """Plot representative examples and save to disk."""
    n_show = len(samples)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.8 * n_show))
    if n_show == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for ax, (idx, sample) in zip(axes, samples):
        raw_img = hf_split[idx]["image"].convert("L")
        ax.imshow(raw_img, cmap="gray", aspect="auto")
        ax.axis("off")
        ax.set_title(
            f"GT:   {sample.reference!r}\n"
            f"PRED: {sample.hypothesis!r}   CER={sample.cer:.3f}",
            loc="left", fontsize=11, pad=4,
        )
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device("cpu")
    cache_dir = cfg.get("dataset", {}).get("cache_dir", None)

    print(f"Loading checkpoint: {args.checkpoint}")
    state = load_checkpoint(args.checkpoint, device=device)
    tokenizer = rebuild_tokenizer_from_checkpoint(state)
    model = CRNN.from_config(tokenizer.vocab_size, state["model_cfg"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    print("Loading test split ...")
    splits = load_iam_splits(cache_dir=cache_dir)
    hf_split = splits["test"]
    
    preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"])
    ds = IAMTorchDataset(hf_split, preprocessor, tokenizer)
    loader = DataLoader(
        ds, batch_size=16, shuffle=False,
        collate_fn=build_collate_fn(pad_value=0.0), num_workers=0,
    )

    decoder = GreedyDecoder(tokenizer)
    all_refs, all_hyps = [], []

    print("Running inference ...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(device)
            logits = model(images)
            hyps = decoder.decode_batch(logits.cpu())
            all_refs.extend(batch["texts"])
            all_hyps.extend(hyps)
            sys.stdout.write(f"\r  batch {batch_idx+1}/{len(loader)}")
            sys.stdout.flush()
    print()

    # 1. Base Metrics
    result = evaluate(all_refs, all_hyps, include_samples=True)
    
    # 2. Aggregate Character Stats using jiwer process_characters
    char_out = jiwer.process_characters(all_refs, all_hyps)
    subs = char_out.substitutions
    dels = char_out.deletions
    inss = char_out.insertions
    hits = char_out.hits
    
    total_ref_chars = subs + dels + hits
    total_errors = subs + dels + inss

    # 3. Analyze by characteristics
    # We will attach the original index to each sample for image fetching
    indexed_samples = list(enumerate(result.samples))
    
    def group_cer(condition_fn):
        group = [s for idx, s in indexed_samples if condition_fn(s.reference)]
        if not group:
            return 0, 0.0
        g_refs = [s.reference for s in group]
        g_hyps = [s.hypothesis for s in group]
        return len(group), jiwer.cer(g_refs, g_hyps)

    groups = {
        "Short (<30 chars)": group_cer(lambda x: len(x) < 30),
        "Medium (30-60 chars)": group_cer(lambda x: 30 <= len(x) <= 60),
        "Long (>60 chars)": group_cer(lambda x: len(x) > 60),
        "Contains Digits": group_cer(lambda x: any(c.isdigit() for c in x)),
        "Contains Punctuation": group_cer(lambda x: any(c in string.punctuation for c in x)),
    }

    # 4. Representative Failures and Successes
    # Filter out empty references just in case
    valid_samples = [(idx, s) for idx, s in indexed_samples if len(s.reference) > 5]
    
    # Sort by CER
    sorted_samples = sorted(valid_samples, key=lambda x: x[1].cer)
    
    successes = sorted_samples[:10]  # Best 10
    failures = sorted_samples[-10:]  # Worst 10
    failures.reverse() # Highest CER first

    # Generate plots
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    plot_examples(successes, hf_split, "Representative Successes (Test Set)", out_dir / "successes.png")
    plot_examples(failures, hf_split, "Representative Failures (Test Set)", out_dir / "failures.png")

    # Generate Markdown Report
    md_path = out_dir / "error_analysis.md"
    with open(md_path, "w") as f:
        f.write("# Baseline Error Analysis\n\n")
        f.write("## 1. Overall Baseline Metrics\n")
        f.write(f"- **Test CER:** {result.cer*100:.2f}%\n")
        f.write(f"- **Test WER:** {result.wer*100:.2f}%\n")
        f.write(f"- **Total Samples:** {result.n_samples}\n\n")

        f.write("## 2. Aggregate Character Statistics\n")
        f.write(f"- **Total Reference Characters:** {total_ref_chars}\n")
        f.write(f"- **Total Errors:** {total_errors}\n")
        f.write(f"- **Substitutions:** {subs} ({(subs/total_ref_chars)*100:.2f}% of ref)\n")
        f.write(f"- **Deletions:** {dels} ({(dels/total_ref_chars)*100:.2f}% of ref)\n")
        f.write(f"- **Insertions:** {inss} ({(inss/total_ref_chars)*100:.2f}% of ref)\n\n")

        f.write("## 3. Analysis by Line Characteristics\n")
        f.write("| Category | Count | CER |\n")
        f.write("|---|---|---|\n")
        for name, (cnt, c_cer) in groups.items():
            f.write(f"| {name} | {cnt} | {c_cer*100:.2f}% |\n")
        
        f.write("\n## 4. Observed Error Patterns\n")
        f.write("*(Analysis summary to be populated by the agent based on output)*\n\n")

        f.write("## 5. Representative Examples\n\n")
        f.write("### 5.1. Top Successes\n")
        for i, (idx, s) in enumerate(successes):
            f.write(f"**[{i+1}]**  \n")
            f.write(f"**GT:** `{s.reference}`  \n")
            f.write(f"**PR:** `{s.hypothesis}`  \n")
            f.write(f"**CER:** {s.cer:.3f}  \n\n")
            
        f.write("### 5.2. Top Failures\n")
        for i, (idx, s) in enumerate(failures):
            f.write(f"**[{i+1}]**  \n")
            f.write(f"**GT:** `{s.reference}`  \n")
            f.write(f"**PR:** `{s.hypothesis}`  \n")
            f.write(f"**CER:** {s.cer:.3f}  \n\n")
            
    print(f"\nAnalysis complete! Report saved to {md_path}")
    print(f"Visualizations saved to outputs/successes.png and outputs/failures.png")


if __name__ == "__main__":
    main()
