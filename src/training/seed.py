"""
seed.py
=======
Reproducibility utility: set random seeds across Python, NumPy, and PyTorch.

Usage
-----
>>> from src.training.seed import set_seed
>>> set_seed(42)
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The seed value to use.

    Notes
    -----
    - For MPS (Apple Silicon), full determinism is not guaranteed without
      setting PYTORCH_ENABLE_MPS_FALLBACK, but setting these seeds still
      significantly reduces non-determinism.
    - torch.backends.cudnn.deterministic is only applied when CUDA is
      available and does not affect MPS or CPU runs.
    - DataLoader worker seeds are not set here; pass worker_init_fn
      if multi-worker determinism is required.
    """
    print(f"[seed] Random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN for CUDA; mild performance cost
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
