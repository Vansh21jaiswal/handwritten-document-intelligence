"""
tokenizer.py
============
Character-level tokenizer for handwritten text recognition.

Design
------
- Vocabulary is built **only** from the training split to avoid data leakage.
- Index 0 is always reserved for the CTC blank token.
- Index 1 is reserved for an <UNK> fallback for unseen characters.
- All other indices follow sorted character order for reproducibility.

CTC convention
--------------
CTC decoding expects the blank to be a dedicated class. We place it at
index 0 so that both training (nn.CTCLoss with blank=0) and decoding
can use a consistent constant.

Encode / Decode
---------------
  encode("hello") → [5, 8, 15, 15, 18]   (example indices)
  decode([5, 8, 15, 15, 18]) → "hello"

Unknown characters are encoded as UNK_IDX and decoded as '' (empty
string) so they are silently dropped during decoding — safe behaviour
for evaluation but logs a warning during training.

Usage
-----
>>> from src.dataset.tokenizer import CharTokenizer
>>> tok = CharTokenizer(blank_token="<BLANK>", unknown_token="<UNK>")
>>> tok.build_vocab(["hello world", "foo bar"])
>>> ids = tok.encode("hello")
>>> text = tok.decode(ids)

Config-driven usage
-------------------
>>> import yaml
>>> cfg = yaml.safe_load(open("configs/default.yaml"))["tokenizer"]
>>> tok = CharTokenizer.from_config(cfg)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

BLANK_IDX: int = 0
UNK_IDX: int = 1


class CharTokenizer:
    """Character-level vocabulary and encode/decode logic.

    Parameters
    ----------
    blank_token : str
        Symbol used as the CTC blank class (always mapped to index 0).
    unknown_token : str
        Symbol used for characters not seen during vocabulary construction.
    """

    def __init__(
        self,
        blank_token: str = "<BLANK>",
        unknown_token: str = "<UNK>",
    ) -> None:
        self.blank_token = blank_token
        self.unknown_token = unknown_token

        # These are populated by build_vocab()
        self._char2idx: Dict[str, int] = {}
        self._idx2char: Dict[int, str] = {}
        self._vocab_built: bool = False

    # ------------------------------------------------------------------
    # Vocabulary construction (training data only)
    # ------------------------------------------------------------------

    def build_vocab(self, texts: Iterable[str]) -> None:
        """Build character vocabulary from an iterable of transcriptions.

        IMPORTANT: call this with training split text ONLY to avoid
        data leakage from validation/test splits into the vocabulary.

        Parameters
        ----------
        texts : Iterable[str]
            Ground-truth transcription strings from the training split.
        """
        chars = set()
        for text in texts:
            chars.update(text)

        # Sorted for reproducibility; blank and unk always come first
        sorted_chars = sorted(chars)

        self._char2idx = {self.blank_token: BLANK_IDX, self.unknown_token: UNK_IDX}
        self._idx2char = {BLANK_IDX: self.blank_token, UNK_IDX: self.unknown_token}

        for char in sorted_chars:
            if char in (self.blank_token, self.unknown_token):
                continue  # Skip if someone used special tokens as literal chars
            idx = len(self._char2idx)
            self._char2idx[char] = idx
            self._idx2char[idx] = char

        self._vocab_built = True
        logger.info(
            "Vocabulary built: %d chars (+blank +unk = %d total indices)",
            len(sorted_chars),
            len(self._char2idx),
        )

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode a transcription string into a list of character indices.

        Unknown characters are mapped to UNK_IDX with a warning.

        Parameters
        ----------
        text : str
            Ground-truth or predicted transcription.

        Returns
        -------
        List[int]
            One integer per character in `text`.
        """
        self._check_built()
        ids: List[int] = []
        for ch in text:
            idx = self._char2idx.get(ch)
            if idx is None:
                logger.warning("Unknown character %r → UNK", ch)
                ids.append(UNK_IDX)
            else:
                ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int], remove_blank: bool = True) -> str:
        """Decode a sequence of character indices back to a string.

        Parameters
        ----------
        ids : Iterable[int]
            Sequence of character indices (e.g. from encode() or CTC output).
        remove_blank : bool
            If True (default), blank tokens are stripped from the output.
            UNK tokens are always dropped silently.

        Returns
        -------
        str
            Decoded transcription string.
        """
        self._check_built()
        chars: List[str] = []
        for idx in ids:
            char = self._idx2char.get(idx, "")
            if idx == BLANK_IDX and remove_blank:
                continue
            if idx == UNK_IDX:
                continue  # drop unknowns silently
            if char:
                chars.append(char)
        return "".join(chars)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of classes including blank and unk."""
        self._check_built()
        return len(self._char2idx)

    @property
    def char2idx(self) -> Dict[str, int]:
        self._check_built()
        return dict(self._char2idx)

    @property
    def idx2char(self) -> Dict[int, str]:
        self._check_built()
        return dict(self._idx2char)

    @property
    def blank_index(self) -> int:
        return BLANK_IDX

    @property
    def unk_index(self) -> int:
        return UNK_IDX

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _check_built(self) -> None:
        if not self._vocab_built:
            raise RuntimeError(
                "Vocabulary has not been built yet. "
                "Call build_vocab(training_texts) first."
            )

    def is_built(self) -> bool:
        return self._vocab_built

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CharTokenizer":
        """Construct from the 'tokenizer' section of default.yaml."""
        return cls(
            blank_token=config.get("blank_token", "<BLANK>"),
            unknown_token=config.get("unknown_token", "<UNK>"),
        )

    def __repr__(self) -> str:
        status = f"vocab_size={self.vocab_size}" if self._vocab_built else "not built"
        return f"CharTokenizer({status}, blank='{self.blank_token}')"
