"""
greedy_decoder.py
=================
CTC greedy decoder for handwritten text recognition.

Algorithm
---------
For each timestep, pick the class with the highest logit (argmax).
Then collapse consecutive duplicate predictions, and remove blank tokens.

Example:
  raw predictions : [a, a, <blank>, b, b, <blank>, c]
  after collapse  : [a, <blank>, b, <blank>, c]
  after blank removal: "abc"

This is the standard CTC greedy decoding procedure described in:
  Graves et al., 2006, "Connectionist Temporal Classification"

Usage
-----
>>> from src.inference.greedy_decoder import GreedyDecoder
>>> decoder = GreedyDecoder(tokenizer)
>>> texts = decoder.decode_batch(logits)   # logits: (T, B, V)
>>> text  = decoder.decode_single(logits[:, 0, :])  # (T, V) for one sample
"""

from __future__ import annotations

from typing import List

import torch

from src.dataset.tokenizer import CharTokenizer, BLANK_IDX


class GreedyDecoder:
    """CTC greedy decoder.

    Parameters
    ----------
    tokenizer : CharTokenizer
        Must have build_vocab() already called.
    """

    def __init__(self, tokenizer: CharTokenizer) -> None:
        self.tokenizer = tokenizer
        self.blank_idx = tokenizer.blank_index  # always 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_batch(self, logits: torch.Tensor) -> List[str]:
        """Decode a batch of logit sequences using CTC greedy decoding.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (T, B, vocab_size). Raw logits (no softmax needed).

        Returns
        -------
        List[str]
            One decoded string per sample in the batch (length B).
        """
        # Greedy: argmax along vocab dimension at each timestep
        # pred: (T, B)
        pred_ids = logits.argmax(dim=2)
        B = pred_ids.shape[1]
        return [self._collapse_and_decode(pred_ids[:, b].tolist()) for b in range(B)]

    def decode_single(self, logits: torch.Tensor) -> str:
        """Decode a single logit sequence.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (T, vocab_size). Raw logits for one sample.

        Returns
        -------
        str
        """
        pred_ids = logits.argmax(dim=1).tolist()  # (T,)
        return self._collapse_and_decode(pred_ids)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _collapse_and_decode(self, ids: List[int]) -> str:
        """Collapse consecutive duplicates, remove blanks, decode to text.

        Parameters
        ----------
        ids : List[int]
            Raw argmax predictions over T timesteps.

        Returns
        -------
        str
            Decoded transcription.
        """
        # Step 1: Collapse consecutive duplicate predictions
        collapsed: List[int] = []
        prev = None
        for idx in ids:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Step 2: Remove blank tokens
        no_blank = [idx for idx in collapsed if idx != self.blank_idx]

        # Step 3: Decode integer IDs → string via tokenizer
        return self.tokenizer.decode(no_blank, remove_blank=False)
