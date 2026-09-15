"""
device.py
=========
Device selection utility.

Selects the best available device:
  1. CUDA  — if a CUDA-capable GPU is detected
  2. MPS   — if running on Apple Silicon with MPS support
  3. CPU   — fallback

Usage
-----
>>> from src.training.device import get_device
>>> device = get_device()          # auto
>>> device = get_device("cuda")    # force CUDA (raises if unavailable)
>>> device = get_device("cpu")     # force CPU
"""

from __future__ import annotations

import os
import torch

# MPS (Apple Silicon) does not currently support _ctc_loss natively.
# Enabling the fallback allows PyTorch to transparently compute it on CPU
# while keeping the rest of the model and tensors on MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_device(preference: str = "auto") -> torch.device:
    """Return the best available torch.device.

    Parameters
    ----------
    preference : str
        "auto"  — detect automatically (cuda > mps > cpu)
        "cuda"  — use CUDA (raises RuntimeError if unavailable)
        "mps"   — use Apple MPS (raises RuntimeError if unavailable)
        "cpu"   — always use CPU

    Returns
    -------
    torch.device
    """
    preference = preference.lower().strip()

    if preference == "cpu":
        device = torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        device = torch.device("cuda")
    elif preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available on this machine.")
        device = torch.device("mps")
    elif preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {preference!r}. "
                         f"Choose from 'auto', 'cuda', 'mps', 'cpu'.")

    print(f"[device] Using: {device}")
    return device
