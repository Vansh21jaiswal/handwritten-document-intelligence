"""
iam_loader.py
=============
Loader for the Teklia/IAM-line dataset hosted on Hugging Face.

Dataset source
--------------
  Name   : Teklia/IAM-line
  URL    : https://huggingface.co/datasets/Teklia/IAM-line
  Origin : IAM Handwriting Database (U.-V. Marti & H. Bunke, 2002)
           University of Bern / HEIA-FR — non-commercial research use only.

Splits
------
  train       — ~6,482 line images used for model training
  validation  — ~976  line images used for hyper-parameter tuning
  test        — ~2,915 line images used for final CER / WER evaluation

Dataset storage
---------------
The dataset is NOT stored in this repository. It is loaded on-demand from
Hugging Face Hub (or from a local HF cache at ~/.cache/huggingface/datasets/).
Do NOT commit dataset images or transcription files to Git.

Each row exposes:
  image      : PIL.Image  — grayscale handwritten line image
  text       : str        — ground-truth transcription

Usage
-----
>>> from src.dataset.iam_loader import load_iam_splits, IAMDataset
>>> splits = load_iam_splits()
>>> print(len(splits["train"]))  # ~6482
>>> sample = splits["train"][0]
>>> print(sample["text"])

For PyTorch training use IAMDataset (see class below).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)

# Canonical HF dataset identifier for the IAM line dataset.
_HF_DATASET_ID = "Teklia/IAM-line"

# Mapping from our canonical split names to the HF split names.
_SPLIT_MAP: Dict[str, str] = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def load_iam_splits(
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> DatasetDict:
    """Load all three IAM-line splits from Hugging Face.

    Parameters
    ----------
    cache_dir : str, optional
        Custom HF cache directory. Defaults to ~/.cache/huggingface/datasets.
    streaming : bool
        If True, datasets are returned as IterableDataset (no local disk write).
        Useful for quick inspection on machines with limited disk space.

    Returns
    -------
    DatasetDict
        A dict-like object with keys "train", "validation", "test".
        Each value is a datasets.Dataset with columns:
            - "image"  : PIL.Image
            - "text"   : str

    Raises
    ------
    ConnectionError
        If the HF Hub is unreachable and no local cache exists.
    """
    logger.info("Loading dataset '%s' from Hugging Face Hub …", _HF_DATASET_ID)

    ds: DatasetDict = load_dataset(
        _HF_DATASET_ID,
        cache_dir=cache_dir,
        streaming=streaming,
    )

    logger.info(
        "Loaded splits — train: %s  val: %s  test: %s",
        "streaming" if streaming else len(ds["train"]),
        "streaming" if streaming else len(ds["validation"]),
        "streaming" if streaming else len(ds["test"]),
    )
    return ds


# ---------------------------------------------------------------------------
# PyTorch-compatible Dataset wrapper (stub — extended in training sprint)
# ---------------------------------------------------------------------------

class IAMDataset:
    """Thin wrapper around a HuggingFace Dataset split.

    Converts raw PIL images and text strings into a format compatible with
    a future PyTorch DataLoader. Preprocessing and tokenisation transforms
    will be injected here during the training sprint.

    Parameters
    ----------
    hf_split : datasets.Dataset
        One split returned by load_iam_splits(), e.g. splits["train"].
    transform : Callable, optional
        Image transform applied to each PIL image (e.g. torchvision transforms).
    """

    def __init__(self, hf_split, transform: Optional[Callable] = None):
        self._split = hf_split
        self.transform = transform

    # ------------------------------------------------------------------
    # Sequence protocol — makes DataLoader happy
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: int) -> Dict:
        row = self._split[idx]
        image = row["image"]
        text = row["text"]

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "text": text}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def texts(self):
        """Return all ground-truth transcription strings as a list."""
        return self._split["text"]

    def sample(self, n: int = 5) -> list:
        """Return n random samples as a list of dicts."""
        import random
        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self[i] for i in indices]
