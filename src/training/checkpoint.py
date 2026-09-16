"""
checkpoint.py
=============
Model checkpointing utilities for the HTR training pipeline.

Saves and loads model + optimizer state, epoch, validation loss,
tokenizer vocabulary, and model configuration — everything needed
to resume training or run inference from a saved state.

Checkpoint format (dict saved via torch.save)
---------------------------------------------
{
    "epoch":         int,
    "val_loss":      float,
    "model_cfg":     dict,          # model section of default.yaml
    "vocab":         dict,          # {char: idx} mapping
    "blank_token":   str,
    "unknown_token": str,
    "model_state":   OrderedDict,   # model.state_dict()
    "optimizer_state": OrderedDict, # optimizer.state_dict()
}

Usage
-----
>>> from src.training.checkpoint import save_checkpoint, load_checkpoint
>>> save_checkpoint(ckpt_dir, "best.pt", model, optimizer, epoch, val_loss, tok, cfg)
>>> state = load_checkpoint("checkpoints/best.pt", model, optimizer, device)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.dataset.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    ckpt_dir: str | Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    tokenizer: CharTokenizer,
    model_cfg: Dict[str, Any],
) -> Path:
    """Save a training checkpoint to disk.

    Parameters
    ----------
    ckpt_dir : str or Path
        Directory where checkpoints are stored. Created if it does not exist.
    filename : str
        Filename within ckpt_dir, e.g. "best.pt" or "latest.pt".
    model : nn.Module
    optimizer : torch.optim.Optimizer
    epoch : int
        The epoch that just completed.
    val_loss : float
        Validation loss achieved at this epoch.
    tokenizer : CharTokenizer
        Must have build_vocab() already called.
    model_cfg : dict
        The 'model' section of default.yaml (for reconstructing the model).

    Returns
    -------
    Path
        Absolute path to the saved checkpoint file.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_cfg": model_cfg,
        "vocab": tokenizer.char2idx,          # {char: idx}
        "blank_token": tokenizer.blank_token,
        "unknown_token": tokenizer.unknown_token,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path = ckpt_dir / filename
    torch.save(state, path)
    logger.info("Checkpoint saved: %s  (epoch=%d  val_loss=%.4f)", path, epoch, val_loss)
    return path


def load_checkpoint(
    ckpt_path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint from disk.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the .pt checkpoint file.
    model : nn.Module, optional
        If provided, loads model_state into it in-place.
    optimizer : torch.optim.Optimizer, optional
        If provided, loads optimizer_state into it in-place.
    device : torch.device, optional
        Map location for loading tensors. Defaults to CPU.

    Returns
    -------
    dict
        The full checkpoint dict (same structure as what was saved).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    map_location = device or torch.device("cpu")
    state = torch.load(ckpt_path, map_location=map_location)

    if model is not None:
        model.load_state_dict(state["model_state"])
        logger.info("Loaded model weights from %s", ckpt_path)

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
        logger.info("Loaded optimizer state from %s", ckpt_path)

    return state


def rebuild_tokenizer_from_checkpoint(state: Dict[str, Any]) -> CharTokenizer:
    """Reconstruct a CharTokenizer from a saved checkpoint state.

    This ensures that the tokenizer used during evaluation exactly
    matches the one used during training, without any data leakage.

    Parameters
    ----------
    state : dict
        Checkpoint dict as returned by load_checkpoint().

    Returns
    -------
    CharTokenizer
        Fully initialised tokenizer with the training vocabulary.
    """
    tokenizer = CharTokenizer(
        blank_token=state["blank_token"],
        unknown_token=state["unknown_token"],
    )
    # Reconstruct vocab from saved {char: idx} mapping
    # We invert to {idx: char} then re-build in sorted-idx order
    char2idx: Dict[str, int] = state["vocab"]
    # Sorted by index to preserve insertion order
    sorted_chars = sorted(char2idx.items(), key=lambda kv: kv[1])

    # Directly populate internal dicts (bypasses build_vocab to preserve exact mapping)
    tokenizer._char2idx = dict(char2idx)
    tokenizer._idx2char = {idx: char for char, idx in char2idx.items()}
    tokenizer._vocab_built = True

    return tokenizer
