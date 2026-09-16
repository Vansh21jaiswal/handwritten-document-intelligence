"""
test_decoder_metrics_checkpoint.py
===================================
Unit tests for:
  - CTC greedy decoding (collapse, blank removal)
  - CER/WER metric computation
  - checkpoint save/load round-trip
  - set_seed reproducibility

All tests use synthetic data — no network access required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.dataset.tokenizer import CharTokenizer, BLANK_IDX
from src.evaluation.metrics import compute_cer, compute_wer, evaluate, per_sample_metrics
from src.inference.greedy_decoder import GreedyDecoder
from src.models.crnn import CRNN
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    rebuild_tokenizer_from_checkpoint,
)
from src.training.seed import set_seed


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.build_vocab(["hello world", "abc def", "the quick brown fox"])
    return tok


@pytest.fixture
def decoder(tokenizer):
    return GreedyDecoder(tokenizer)


@pytest.fixture
def tiny_model(tokenizer):
    return CRNN(
        vocab_size=tokenizer.vocab_size,
        cnn_channels=[8, 16, 32, 64],
        rnn_hidden=32,
        rnn_layers=1,
        cnn_dropout=0.0,
        rnn_dropout=0.0,
    )


# ────────────────────────────────────────────────────────────────────────────
# Greedy decoder tests
# ────────────────────────────────────────────────────────────────────────────

class TestGreedyDecoder:

    def _logits_from_ids(self, ids: list, vocab_size: int) -> torch.Tensor:
        """Create one-hot logits for a sequence of IDs. Shape: (T, 1, V)."""
        T = len(ids)
        logits = torch.full((T, 1, vocab_size), -100.0)
        for t, idx in enumerate(ids):
            logits[t, 0, idx] = 100.0
        return logits

    def test_perfect_decoding_no_repeats(self, decoder, tokenizer):
        """Encoding then decoding a word with no consecutive repeated chars."""
        # "the" has no consecutive duplicate chars, so greedy CTC decoding
        # from a clean one-hot sequence must reproduce it exactly.
        text = "the"
        ids = tokenizer.encode(text)
        logits = self._logits_from_ids(ids, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert decoded[0] == text

    def test_repeated_chars_require_blank_separator(self, decoder, tokenizer):
        """CTC: to emit 'hello' (two l's), a blank must separate the l's."""
        # Without a blank, consecutive l's collapse → "helo"  (correct CTC)
        h = tokenizer.char2idx["h"]
        e = tokenizer.char2idx["e"]
        l_ = tokenizer.char2idx["l"]
        o = tokenizer.char2idx["o"]
        ids_no_blank = [h, e, l_, l_, o]
        logits = self._logits_from_ids(ids_no_blank, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert decoded[0] == "helo"        # expected collapse behaviour

        # With a blank between the l's, both are preserved → "hello"
        ids_with_blank = [h, e, l_, BLANK_IDX, l_, o]
        logits2 = self._logits_from_ids(ids_with_blank, tokenizer.vocab_size)
        decoded2 = decoder.decode_batch(logits2)
        assert decoded2[0] == "hello"

    def test_collapse_consecutive_duplicates(self, decoder, tokenizer):
        """Consecutive duplicate IDs must be collapsed to one."""
        h_id = tokenizer.char2idx["h"]
        ids = [h_id, h_id, h_id]   # "h h h" → "h"
        logits = self._logits_from_ids(ids, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert decoded[0] == "h"

    def test_blank_removal(self, decoder, tokenizer):
        """Blanks must be removed from final output."""
        h_id = tokenizer.char2idx["h"]
        ids = [BLANK_IDX, h_id, BLANK_IDX]
        logits = self._logits_from_ids(ids, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert decoded[0] == "h"

    def test_ctc_collapse_example(self, decoder, tokenizer):
        """
        Classic CTC example: [a, a, blank, b, b, blank, c] → "abc"
        """
        a = tokenizer.char2idx.get("a")
        b = tokenizer.char2idx.get("b")
        c = tokenizer.char2idx.get("c")
        if a is None or b is None or c is None:
            pytest.skip("chars a/b/c not in fixture vocabulary")

        ids = [a, a, BLANK_IDX, b, b, BLANK_IDX, c]
        logits = self._logits_from_ids(ids, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert decoded[0] == "abc"

    def test_all_blanks_gives_empty_string(self, decoder, tokenizer):
        ids = [BLANK_IDX, BLANK_IDX, BLANK_IDX]
        logits = self._logits_from_ids(ids, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert decoded[0] == ""

    def test_batch_size_matches(self, decoder, tokenizer):
        B = 4
        T = 10
        logits = torch.randn(T, B, tokenizer.vocab_size)
        decoded = decoder.decode_batch(logits)
        assert len(decoded) == B

    def test_decode_single(self, decoder, tokenizer):
        """decode_single works on (T, V) tensors."""
        h_id = tokenizer.char2idx["h"]
        logits_1d = torch.full((5, tokenizer.vocab_size), -100.0)
        logits_1d[:, h_id] = 100.0
        result = decoder.decode_single(logits_1d)
        assert result == "h"


# ────────────────────────────────────────────────────────────────────────────
# CER / WER tests
# ────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_perfect_cer(self):
        assert compute_cer(["hello"], ["hello"]) == pytest.approx(0.0)

    def test_perfect_wer(self):
        assert compute_wer(["hello world"], ["hello world"]) == pytest.approx(0.0)

    def test_cer_finite(self):
        cer = compute_cer(["hello world"], ["helo wrld"])
        assert 0.0 < cer <= 1.5

    def test_wer_finite(self):
        wer = compute_wer(["hello world"], ["helo world"])
        assert 0.0 < wer <= 1.5

    def test_cer_empty_input(self):
        assert compute_cer([], []) == 0.0

    def test_wer_empty_input(self):
        assert compute_wer([], []) == 0.0

    def test_evaluate_returns_correct_n(self):
        refs = ["hello", "world", "foo"]
        hyps = ["hello", "wrld", "foo"]
        result = evaluate(refs, hyps)
        assert result.n_samples == 3

    def test_evaluate_includes_samples(self):
        refs = ["hello", "world"]
        hyps = ["hello", "wrld"]
        result = evaluate(refs, hyps, include_samples=True)
        assert len(result.samples) == 2

    def test_per_sample_metrics_perfect(self):
        samples = per_sample_metrics(["hello"], ["hello"])
        assert samples[0].cer == pytest.approx(0.0)
        assert samples[0].wer == pytest.approx(0.0)

    def test_per_sample_fields(self):
        samples = per_sample_metrics(["hello world"], ["hello wrld"])
        s = samples[0]
        assert s.reference == "hello world"
        assert s.hypothesis == "hello wrld"
        assert hasattr(s, "cer")
        assert hasattr(s, "wer")


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint tests
# ────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:

    def test_save_and_load_roundtrip(self, tiny_model, tokenizer):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        cfg = {
            "cnn_channels": [8, 16, 32, 64],
            "cnn_kernel_size": 3,
            "cnn_dropout": 0.0,
            "rnn_hidden": 32,
            "rnn_layers": 1,
            "rnn_dropout": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                tmpdir, "test.pt",
                tiny_model, optimizer, epoch=1, val_loss=3.14,
                tokenizer=tokenizer, model_cfg=cfg,
            )
            assert path.exists()
            state = load_checkpoint(path)

        assert state["epoch"] == 1
        assert state["val_loss"] == pytest.approx(3.14)

    def test_model_weights_restored(self, tiny_model, tokenizer):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        cfg = {
            "cnn_channels": [8, 16, 32, 64],
            "cnn_kernel_size": 3,
            "cnn_dropout": 0.0,
            "rnn_hidden": 32,
            "rnn_layers": 1,
            "rnn_dropout": 0.0,
        }

        # Snapshot weights before save
        before = {n: p.data.clone() for n, p in tiny_model.named_parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                tmpdir, "test.pt",
                tiny_model, optimizer, epoch=1, val_loss=0.5,
                tokenizer=tokenizer, model_cfg=cfg,
            )
            # Create fresh model and load
            new_model = CRNN(
                vocab_size=tokenizer.vocab_size,
                cnn_channels=[8, 16, 32, 64],
                rnn_hidden=32, rnn_layers=1, cnn_dropout=0.0, rnn_dropout=0.0,
            )
            load_checkpoint(path, model=new_model)

        for name, param in new_model.named_parameters():
            assert torch.equal(before[name], param.data), f"Mismatch in {name}"

    def test_tokenizer_rebuilt_from_checkpoint(self, tiny_model, tokenizer):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        cfg = {"cnn_channels": [8, 16, 32, 64], "cnn_kernel_size": 3,
               "cnn_dropout": 0.0, "rnn_hidden": 32, "rnn_layers": 1, "rnn_dropout": 0.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                tmpdir, "test.pt",
                tiny_model, optimizer, 1, 0.5, tokenizer, cfg,
            )
            state = load_checkpoint(path)
            rebuilt = rebuild_tokenizer_from_checkpoint(state)

        assert rebuilt.vocab_size == tokenizer.vocab_size
        assert rebuilt.blank_index == tokenizer.blank_index
        assert rebuilt.char2idx == tokenizer.char2idx

    def test_missing_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/model.pt")


# ────────────────────────────────────────────────────────────────────────────
# Seed tests
# ────────────────────────────────────────────────────────────────────────────

class TestSeed:

    def test_same_seed_same_tensor(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_different_seed_different_tensor(self):
        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        assert not torch.equal(t1, t2)
