"""
ctc_loss.py
===========
CTC loss wrapper for the HTR training pipeline.

Why a wrapper?
--------------
PyTorch's nn.CTCLoss requires careful argument ordering and the blank index
must match the tokenizer's BLANK_IDX (always 0 in this project).
This module centralises that contract so neither the training loop nor the
model need to be aware of implementation details.

Usage
-----
>>> from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
>>> criterion = build_ctc_loss(blank=tokenizer.blank_index)
>>> loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def build_ctc_loss(blank: int = 0) -> nn.CTCLoss:
    """Create a CTCLoss instance with the correct blank index.

    Parameters
    ----------
    blank : int
        The CTC blank class index. Must equal tokenizer.blank_index (always 0).
        zero_infinity=True silences -inf losses that can occur on very short
        sequences or badly initialised models during the first few iterations.

    Returns
    -------
    nn.CTCLoss
    """
    return nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)


def compute_ctc_loss(
    criterion: nn.CTCLoss,
    logits: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    """Apply log-softmax and compute CTC loss.

    Parameters
    ----------
    criterion : nn.CTCLoss
        As returned by build_ctc_loss().
    logits : Tensor
        Raw model output, shape (T, B, vocab_size). No softmax applied yet.
    targets : Tensor
        Flat concatenated label indices, shape (sum_T,), dtype=long.
        As produced by the collate function.
    input_lengths : Tensor
        Number of valid time steps per sample, shape (B,), dtype=long.
        Computed by model.compute_input_lengths(image_widths).
    target_lengths : Tensor
        Label sequence length per sample, shape (B,), dtype=long.

    Returns
    -------
    Tensor
        Scalar loss value.

    Notes
    -----
    CTCLoss requires log-probabilities, not raw logits. We apply
    log_softmax here so the model's forward() can stay clean.
    """
    log_probs = logits.log_softmax(dim=2)  # (T, B, vocab_size)
    
    # MPS does not currently support native CTC loss.
    # We must explicitly move tensors to CPU to compute the loss,
    # then move the scalar result back to MPS to preserve the computation graph.
    is_mps = log_probs.device.type == "mps"
    if is_mps:
        log_probs = log_probs.to("cpu")
        targets = targets.to("cpu")
        input_lengths = input_lengths.to("cpu")
        target_lengths = target_lengths.to("cpu")
        
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    
    if is_mps:
        loss = loss.to("mps")
        
    return loss
