"""
Dataset sub-package.

Handles dataset discovery, loading, splitting,
and PyTorch Dataset / DataLoader construction.

Public API
----------
load_iam_splits()   -> dict[str, datasets.Dataset]
IAMDataset          -> torch-compatible wrapper (to be extended in training sprint)
"""

from src.dataset.iam_loader import load_iam_splits, IAMDataset

__all__ = ["load_iam_splits", "IAMDataset"]
