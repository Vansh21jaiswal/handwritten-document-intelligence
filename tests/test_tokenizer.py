"""
test_tokenizer.py
=================
Unit tests for the character-level tokenizer.

No IAM dataset download required — uses small synthetic text.
"""

import pytest

from src.dataset.tokenizer import CharTokenizer, BLANK_IDX, UNK_IDX


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

TRAIN_TEXTS = [
    "hello world",
    "foo bar baz",
    "the quick brown fox",
]


@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.build_vocab(TRAIN_TEXTS)
    return tok


# ────────────────────────────────────────────────────────────────────────────
# Vocabulary construction
# ────────────────────────────────────────────────────────────────────────────

class TestVocabConstruction:

    def test_blank_is_index_zero(self, tokenizer):
        assert tokenizer.char2idx[tokenizer.blank_token] == BLANK_IDX == 0

    def test_unk_is_index_one(self, tokenizer):
        assert tokenizer.char2idx[tokenizer.unknown_token] == UNK_IDX == 1

    def test_vocab_contains_all_training_chars(self, tokenizer):
        expected = set("".join(TRAIN_TEXTS))
        for ch in expected:
            assert ch in tokenizer.char2idx, f"'{ch}' missing from vocab"

    def test_vocab_size_correct(self, tokenizer):
        unique_chars = set("".join(TRAIN_TEXTS))
        # +2 for blank and unk
        assert tokenizer.vocab_size == len(unique_chars) + 2

    def test_vocab_sorted_reproducible(self):
        tok1 = CharTokenizer()
        tok2 = CharTokenizer()
        tok1.build_vocab(["abc", "xyz"])
        tok2.build_vocab(["xyz", "abc"])  # different order
        assert tok1.char2idx == tok2.char2idx, "Vocab should be order-independent"

    def test_training_only_no_test_leakage(self, tokenizer):
        """Characters only present in test/val should NOT be in vocab."""
        test_only_char = "€"  # unlikely to be in TRAIN_TEXTS
        assert test_only_char not in tokenizer.char2idx


# ────────────────────────────────────────────────────────────────────────────
# Encode
# ────────────────────────────────────────────────────────────────────────────

class TestEncode:

    def test_encode_returns_list_of_ints(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_length_matches_text(self, tokenizer):
        text = "hello"
        ids = tokenizer.encode(text)
        assert len(ids) == len(text)

    def test_encode_known_chars(self, tokenizer):
        text = "hello"
        ids = tokenizer.encode(text)
        for ch, idx in zip(text, ids):
            assert tokenizer.char2idx[ch] == idx

    def test_encode_unknown_char_returns_unk(self, tokenizer):
        ids = tokenizer.encode("€€€")
        assert ids == [UNK_IDX, UNK_IDX, UNK_IDX]

    def test_encode_empty_string(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_encode_mixed_known_unknown(self, tokenizer):
        ids = tokenizer.encode("h€llo")
        assert ids[0] == tokenizer.char2idx["h"]
        assert ids[1] == UNK_IDX
        assert ids[2] == tokenizer.char2idx["l"]


# ────────────────────────────────────────────────────────────────────────────
# Decode
# ────────────────────────────────────────────────────────────────────────────

class TestDecode:

    def test_decode_roundtrip(self, tokenizer):
        text = "hello"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_decode_strips_blank_by_default(self, tokenizer):
        ids = [BLANK_IDX, tokenizer.char2idx["h"], BLANK_IDX]
        result = tokenizer.decode(ids)
        assert result == "h"

    def test_decode_keeps_blank_when_requested(self, tokenizer):
        ids = [BLANK_IDX, tokenizer.char2idx["h"]]
        result = tokenizer.decode(ids, remove_blank=False)
        assert tokenizer.blank_token in result or result.startswith("<")

    def test_decode_drops_unk_silently(self, tokenizer):
        ids = [UNK_IDX, tokenizer.char2idx["h"], UNK_IDX]
        result = tokenizer.decode(ids)
        assert result == "h"

    def test_decode_empty_sequence(self, tokenizer):
        assert tokenizer.decode([]) == ""

    def test_decode_all_blanks(self, tokenizer):
        assert tokenizer.decode([BLANK_IDX, BLANK_IDX, BLANK_IDX]) == ""


# ────────────────────────────────────────────────────────────────────────────
# Error handling
# ────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_encode_before_build_raises(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="build_vocab"):
            tok.encode("hello")

    def test_decode_before_build_raises(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="build_vocab"):
            tok.decode([1, 2, 3])

    def test_vocab_size_before_build_raises(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="build_vocab"):
            _ = tok.vocab_size


# ────────────────────────────────────────────────────────────────────────────
# Config factory
# ────────────────────────────────────────────────────────────────────────────

class TestFromConfig:

    def test_from_config(self):
        cfg = {"blank_token": "<B>", "unknown_token": "<U>"}
        tok = CharTokenizer.from_config(cfg)
        assert tok.blank_token == "<B>"
        assert tok.unknown_token == "<U>"

    def test_from_config_defaults(self):
        tok = CharTokenizer.from_config({})
        assert tok.blank_token == "<BLANK>"
        assert tok.unknown_token == "<UNK>"
