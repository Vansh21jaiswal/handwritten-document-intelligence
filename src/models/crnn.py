"""
crnn.py
=======
CNN-BiLSTM-CTC model for offline handwritten text recognition.

Architecture overview
---------------------

  Input:  (B, 1, H=128, W)      — grayscale line image, variable width

  ┌─────────────────────────────┐
  │  CNN Feature Extractor      │  4 conv blocks (Conv → BN → ReLU → Pool)
  │  Output: (B, C, H', W')     │  H' is collapsed; W' is the time axis
  └─────────────────────────────┘
              ↓
  ┌─────────────────────────────┐
  │  Height Collapse            │  AdaptiveAvgPool → squeeze → permute
  │  Output: (W', B, C)         │  W' timesteps, each of dimension C
  └─────────────────────────────┘
              ↓
  ┌─────────────────────────────┐
  │  BiLSTM                     │  2 stacked bidirectional LSTM layers
  │  Output: (W', B, 2·H)       │  H = rnn_hidden per direction
  └─────────────────────────────┘
              ↓
  ┌─────────────────────────────┐
  │  Linear Classifier          │  Projects to vocab_size
  │  Output: (W', B, vocab_size)│  Raw logits (no softmax) for CTCLoss
  └─────────────────────────────┘

CTC training
------------
  loss = nn.CTCLoss(blank=tokenizer.blank_index)(
      log_probs,       # (W', B, vocab_size)  — log-softmax applied internally
      targets,         # (sum_T,)             — concatenated flat labels
      input_lengths,   # (B,)                 — W' per sample (after padding mask)
      target_lengths,  # (B,)                 — label length per sample
  )

  input_lengths = image_widths // cnn_width_reduction
  where cnn_width_reduction = product of all width-direction strides in CNN.

Usage
-----
>>> from src.models.crnn import CRNN
>>> model = CRNN(vocab_size=82, cfg=cfg["model"])
>>> logits = model(images)                # (W', B, vocab_size)
>>> log_probs = logits.log_softmax(dim=2)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────
# CNN block helper
# ────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Single Conv → BatchNorm → ReLU → MaxPool block.

    Pooling strategy:
      - Blocks 0–1: pool (2, 2) — reduce both H and W by ×2
      - Block  2  : pool (2, 1) — reduce H only; preserve W resolution
      - Block  3  : pool (2, 1) — reduce H only; preserve W resolution

    This collapses the 128-px height to 128/16 = 8 rows while keeping
    the width (time) resolution as high as possible, giving the BiLSTM
    more time steps to work with.

    Overall CNN width reduction factor: 2 × 2 = 4  (from blocks 0–1).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        pool_size: tuple = (2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


# ────────────────────────────────────────────────────────────────────────────
# Main model
# ────────────────────────────────────────────────────────────────────────────

class CRNN(nn.Module):
    """CNN-BiLSTM-CTC model for handwritten text recognition.

    Parameters
    ----------
    vocab_size : int
        Total number of output classes including the CTC blank and <UNK>.
        Must match tokenizer.vocab_size.
    cnn_channels : list of int
        Output channel counts for the 4 CNN blocks.
        Default: [32, 64, 128, 256]
    cnn_kernel_size : int
        Convolution kernel size (same for all blocks). Default: 3.
    cnn_dropout : float
        Dropout probability applied after each CNN block. Default: 0.1.
    rnn_hidden : int
        Number of hidden units per LSTM direction. Default: 256.
    rnn_layers : int
        Number of stacked BiLSTM layers. Default: 2.
    rnn_dropout : float
        Dropout between BiLSTM layers (ignored if rnn_layers==1). Default: 0.3.

    Notes
    -----
    CNN width reduction factor = 4  (two ×2 MaxPool layers with stride 2 in W).
    To compute input_lengths for CTCLoss:
        input_lengths = image_widths // self.cnn_width_reduction
    """

    # Total factor by which the CNN reduces the image width.
    # Determined by pool_size choices in _ConvBlock.
    cnn_width_reduction: int = 4

    def __init__(
        self,
        vocab_size: int,
        cnn_channels: List[int] = None,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.1,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]
        assert len(cnn_channels) == 4, "cnn_channels must have exactly 4 elements"

        # ── CNN Feature Extractor ──────────────────────────────────────
        # Pool sizes chosen to:
        #   - Aggressively reduce H (128 → 8 after 4 blocks: ÷2 ÷2 ÷2 ÷2)
        #   - Reduce W by ×4 total (only first two blocks pool in W)
        in_channels = [1] + cnn_channels[:-1]
        pool_sizes = [(2, 2), (2, 2), (2, 1), (2, 1)]

        self.cnn = nn.Sequential(*[
            _ConvBlock(in_ch, out_ch, cnn_kernel_size, pool, cnn_dropout)
            for in_ch, out_ch, pool in zip(in_channels, cnn_channels, pool_sizes)
        ])

        # ── Height Collapse ────────────────────────────────────────────
        # After CNN: (B, C=cnn_channels[-1], H'=8, W')
        # AdaptiveAvgPool2d(1, None) collapses H' → 1 while keeping W'.
        # Result after squeeze: (B, C, W')
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        cnn_out_ch = cnn_channels[-1]

        # ── BiLSTM Sequence Model ──────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=cnn_out_ch,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,       # expects (T, B, C)
            bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )

        # ── Linear Classifier ──────────────────────────────────────────
        # 2 * rnn_hidden because bidirectional concatenates forward + backward
        self.classifier = nn.Linear(rnn_hidden * 2, vocab_size)

        self.vocab_size = vocab_size
        self.rnn_hidden = rnn_hidden

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W). H must be 128 for the default CNN config.

        Returns
        -------
        torch.Tensor
            Raw logits of shape (T, B, vocab_size) where T = W // 4.
            Apply log_softmax(dim=2) before passing to nn.CTCLoss.
        """
        # CNN: (B, 1, H, W) → (B, C, H', W')
        x = self.cnn(x)

        # Height collapse: (B, C, H', W') → (B, C, 1, W') → (B, C, W')
        x = self.height_pool(x).squeeze(2)

        # Permute for LSTM: (B, C, W') → (W', B, C)
        x = x.permute(2, 0, 1)

        # BiLSTM: (W', B, C) → (W', B, 2·rnn_hidden)
        x, _ = self.rnn(x)

        # Linear: (W', B, 2·rnn_hidden) → (W', B, vocab_size)
        x = self.classifier(x)

        return x  # raw logits — caller applies log_softmax for CTCLoss

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compute_input_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        """Convert image pixel widths to CTC input sequence lengths.

        Parameters
        ----------
        image_widths : torch.Tensor
            Shape (B,), dtype=long. Original image widths before padding.

        Returns
        -------
        torch.Tensor
            Shape (B,), dtype=long. Number of time steps the model produces
            for each image in the batch, accounting for CNN width reduction.
        """
        return (image_widths // self.cnn_width_reduction).long()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, vocab_size: int, cfg: Dict[str, Any]) -> "CRNN":
        """Construct from the 'model' section of default.yaml.

        Parameters
        ----------
        vocab_size : int
            Must equal tokenizer.vocab_size.
        cfg : dict
            The model sub-dict from default.yaml.
        """
        return cls(
            vocab_size=vocab_size,
            cnn_channels=cfg.get("cnn_channels", [32, 64, 128, 256]),
            cnn_kernel_size=cfg.get("cnn_kernel_size", 3),
            cnn_dropout=cfg.get("cnn_dropout", 0.1),
            rnn_hidden=cfg.get("rnn_hidden", 256),
            rnn_layers=cfg.get("rnn_layers", 2),
            rnn_dropout=cfg.get("rnn_dropout", 0.3),
        )

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return (
            f"CRNN(vocab={self.vocab_size}, "
            f"rnn_hidden={self.rnn_hidden}, "
            f"params={params:,})"
        )
