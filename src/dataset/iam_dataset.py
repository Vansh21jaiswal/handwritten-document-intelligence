"""
iam_dataset.py
==============
PyTorch Dataset and collate function for the IAM Handwriting Database.

This module wires together:
  - ImagePreprocessor  (src.preprocessing)
  - CharTokenizer      (src.dataset.tokenizer)
  - HF IAM-line split  (src.dataset.iam_loader)

into a proper torch.utils.data.Dataset / DataLoader pipeline.

Classes
-------
IAMTorchDataset
    torch.utils.data.Dataset wrapping one HF split. Returns
    processed image tensors and encoded label tensors.

Functions
---------
build_collate_fn(pad_value)
    Returns a collate function that pads variable-width images along
    the W dimension and pads label sequences — designed for CTC training.

Usage
-----
>>> from src.dataset.iam_loader   import load_iam_splits
>>> from src.dataset.tokenizer    import CharTokenizer
>>> from src.preprocessing        import ImagePreprocessor
>>> from src.dataset.iam_dataset  import IAMTorchDataset, build_collate_fn
>>> from torch.utils.data         import DataLoader

>>> splits    = load_iam_splits()
>>> tokenizer = CharTokenizer()
>>> tokenizer.build_vocab(splits["train"]["text"])   # train texts only!
>>> preprocessor = ImagePreprocessor(target_height=128)

>>> train_ds = IAMTorchDataset(splits["train"], preprocessor, tokenizer)
>>> loader   = DataLoader(
...     train_ds,
...     batch_size=16,
...     collate_fn=build_collate_fn(pad_value=0.0),
...     shuffle=True,
... )
>>> batch = next(iter(loader))
>>> batch["images"].shape       # (B, 1, 128, W_max)
>>> batch["targets"]            # (total_label_chars,)  1-D for CTCLoss
>>> batch["target_lengths"]     # (B,)
>>> batch["image_widths"]       # (B,)  original widths before padding
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.preprocessing.image_transforms import ImagePreprocessor
from src.dataset.tokenizer import CharTokenizer


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class IAMTorchDataset(Dataset):
    """PyTorch Dataset for one split of the IAM-line dataset.

    Each item returned by __getitem__ is a dict with:
        "image"         : torch.Tensor  shape (1, H, W)  float32
        "target"        : torch.Tensor  shape (T,)        int64 — encoded label
        "target_length" : int           — T (number of characters)
        "text"          : str           — raw transcription (for eval / debug)
        "image_width"   : int           — W (original width, pre-padding)

    Parameters
    ----------
    hf_split : datasets.Dataset
        One split returned by load_iam_splits().
    preprocessor : ImagePreprocessor
        Image preprocessing pipeline.
    tokenizer : CharTokenizer
        Must have build_vocab() already called on training texts.
    """

    def __init__(
        self,
        hf_split,
        preprocessor: ImagePreprocessor,
        tokenizer: CharTokenizer,
    ) -> None:
        if not tokenizer.is_built():
            raise RuntimeError(
                "Tokenizer vocabulary has not been built. "
                "Call tokenizer.build_vocab(train_texts) before constructing IAMTorchDataset."
            )
        self._split = hf_split
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._split[idx]
        pil_image = row["image"]
        text = row["text"]

        # Image → normalised tensor (1, H, W)
        image_tensor = self.preprocessor(pil_image)
        image_width = image_tensor.shape[2]  # W dimension

        # Text → encoded integer tensor
        encoded = self.tokenizer.encode(text)
        target_tensor = torch.tensor(encoded, dtype=torch.long)

        return {
            "image": image_tensor,          # (1, H, W)
            "target": target_tensor,        # (T,)
            "target_length": len(encoded),  # scalar int
            "text": text,                   # raw string
            "image_width": image_width,     # scalar int
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


# ────────────────────────────────────────────────────────────────────────────
# Collate function
# ────────────────────────────────────────────────────────────────────────────

def build_collate_fn(pad_value: float = 0.0) -> Callable:
    """Return a collate function that handles variable-width images.

    The returned function batches a list of items from IAMTorchDataset
    into a dict suitable for CTC training:

    Output batch keys
    -----------------
    "images"         : Tensor (B, 1, H, W_max) — width-padded image batch
    "targets"        : Tensor (sum_of_T,)       — concatenated flat labels
                       (the format expected by nn.CTCLoss)
    "target_lengths" : Tensor (B,)              — label length per sample
    "image_widths"   : Tensor (B,)              — original width before padding
    "texts"          : List[str]                — raw transcriptions

    Parameters
    ----------
    pad_value : float
        Value used to pad image tensors along the width axis. Default 0.0
        corresponds to the normalised background, which (mean=0.5, std=0.5)
        maps to a mid-grey pixel. Using 0.0 is a common convention.

    Notes
    -----
    Height is never padded — all images share the same fixed height after
    preprocessing. Only the width dimension is padded to the maximum width
    in the batch. Individual images are never rescaled or distorted.

    The targets tensor is 1-D (concatenated) as required by torch.nn.CTCLoss.
    Use target_lengths to split it back per sample during loss computation.
    """

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [item["image"] for item in batch]                    # list of (1,H,W_i)
        targets = [item["target"] for item in batch]                  # list of (T_i,)
        target_lengths = torch.tensor(
            [item["target_length"] for item in batch], dtype=torch.long
        )
        image_widths = torch.tensor(
            [item["image_width"] for item in batch], dtype=torch.long
        )
        texts = [item["text"] for item in batch]

        # ── Pad images along W to max width in batch ──────────────────
        max_width = max(img.shape[2] for img in images)
        height = images[0].shape[1]
        padded_images = torch.full(
            (len(images), 1, height, max_width),
            fill_value=pad_value,
            dtype=torch.float32,
        )
        for i, img in enumerate(images):
            w = img.shape[2]
            padded_images[i, :, :, :w] = img

        # ── Concatenate targets into a single 1-D tensor ──────────────
        # nn.CTCLoss expects targets as (N, S) or concatenated (sum_T,)
        flat_targets = torch.cat(targets, dim=0)  # (sum_T,)

        return {
            "images": padded_images,           # (B, 1, H, W_max)
            "targets": flat_targets,           # (sum_T,)
            "target_lengths": target_lengths,  # (B,)
            "image_widths": image_widths,      # (B,)
            "texts": texts,                    # List[str]
        }

    return collate
