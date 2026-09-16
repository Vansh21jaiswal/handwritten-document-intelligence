"""
Training sub-package.

Public API
----------
get_device           — device selection (cuda > mps > cpu)
set_seed             — reproducible seed across Python/NumPy/PyTorch
build_ctc_loss       — creates nn.CTCLoss(blank=0, zero_infinity=True)
compute_ctc_loss     — applies log_softmax then CTCLoss (MPS-safe)
save_checkpoint      — save model+optimizer+tokenizer state
load_checkpoint      — load checkpoint from disk
rebuild_tokenizer_from_checkpoint — restore tokenizer vocab from checkpoint
build_data_pipeline  — construct tokenizer, datasets, dataloaders from HF splits
train_one_epoch      — one training pass over a DataLoader
validate_epoch       — one validation pass returning (loss, CER, WER)
run_training         — full training orchestrator using HF local cache
"""

from src.training.device import get_device
from src.training.seed import set_seed
from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    rebuild_tokenizer_from_checkpoint,
)
from src.training.trainer import (
    build_data_pipeline,
    train_one_epoch,
    validate_epoch,
    run_training,
)

__all__ = [
    "get_device",
    "set_seed",
    "build_ctc_loss",
    "compute_ctc_loss",
    "save_checkpoint",
    "load_checkpoint",
    "rebuild_tokenizer_from_checkpoint",
    "build_data_pipeline",
    "train_one_epoch",
    "validate_epoch",
    "run_training",
]
