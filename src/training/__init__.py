"""
Training sub-package.

Public API
----------
get_device          — device selection (cuda > mps > cpu)
build_ctc_loss      — creates nn.CTCLoss(blank=0, zero_infinity=True)
compute_ctc_loss    — applies log_softmax then CTCLoss
train_one_epoch     — one training pass over a DataLoader
validate            — one validation pass over a DataLoader
run_training        — full training orchestrator
"""

from src.training.device import get_device
from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
from src.training.trainer import train_one_epoch, validate, run_training

__all__ = [
    "get_device",
    "build_ctc_loss",
    "compute_ctc_loss",
    "train_one_epoch",
    "validate",
    "run_training",
]
