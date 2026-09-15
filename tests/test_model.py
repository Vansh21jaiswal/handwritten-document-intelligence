"""
test_model.py
=============
Unit tests for the CRNN model.

All tests use synthetic tensors — no IAM dataset or network access required.
"""

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.crnn import CRNN
from src.dataset.tokenizer import CharTokenizer, BLANK_IDX
from src.training.ctc_loss import build_ctc_loss, compute_ctc_loss
from src.training.device import get_device


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 50
IMG_HEIGHT = 128


@pytest.fixture
def model():
    return CRNN(
        vocab_size=VOCAB_SIZE,
        cnn_channels=[16, 32, 64, 128],
        rnn_hidden=64,
        rnn_layers=1,
        cnn_dropout=0.0,
        rnn_dropout=0.0,
    )


def make_batch(B: int, W: int, H: int = IMG_HEIGHT) -> torch.Tensor:
    """Synthetic (B, 1, H, W) float tensor."""
    return torch.rand(B, 1, H, W)


# ────────────────────────────────────────────────────────────────────────────
# Forward pass shape tests
# ────────────────────────────────────────────────────────────────────────────

class TestForwardShape:

    def test_output_ndim(self, model):
        x = make_batch(2, 512)
        out = model(x)
        assert out.ndim == 3, f"Expected 3-D output (T, B, V), got {out.ndim}-D"

    def test_output_batch_dim(self, model):
        B = 3
        x = make_batch(B, 512)
        out = model(x)
        assert out.shape[1] == B

    def test_output_vocab_dim(self, model):
        x = make_batch(2, 512)
        out = model(x)
        assert out.shape[2] == VOCAB_SIZE

    def test_time_dim_equals_width_div_4(self, model):
        """T should equal W // cnn_width_reduction (== 4)."""
        W = 400
        x = make_batch(1, W)
        out = model(x)
        expected_T = W // CRNN.cnn_width_reduction
        assert out.shape[0] == expected_T, (
            f"Expected T={expected_T}, got T={out.shape[0]}"
        )

    def test_variable_width_batch_size_1(self, model):
        """Model must handle different widths without error."""
        for W in [128, 256, 512, 1024, 2048]:
            x = make_batch(1, W)
            out = model(x)
            assert out.shape[0] == W // 4

    def test_variable_width_batch(self, model):
        """Batch of same-width images (after collate padding)."""
        x = make_batch(4, 800)
        out = model(x)
        assert out.shape == (200, 4, VOCAB_SIZE)


# ────────────────────────────────────────────────────────────────────────────
# compute_input_lengths
# ────────────────────────────────────────────────────────────────────────────

class TestInputLengths:

    def test_compute_input_lengths(self, model):
        widths = torch.tensor([400, 800, 1200])
        lengths = model.compute_input_lengths(widths)
        expected = torch.tensor([100, 200, 300])
        assert torch.equal(lengths, expected)

    def test_input_lengths_dtype(self, model):
        widths = torch.tensor([400, 800])
        lengths = model.compute_input_lengths(widths)
        assert lengths.dtype == torch.long


# ────────────────────────────────────────────────────────────────────────────
# CTC loss integration
# ────────────────────────────────────────────────────────────────────────────

class TestCTCLoss:

    def _make_valid_batch(self, model, B=2, W=512, label_len=5):
        """Create a valid batch guaranteed to satisfy CTC constraint T >= label_len."""
        x = make_batch(B, W)
        logits = model(x)                         # (T, B, V)
        T = logits.shape[0]

        targets = torch.randint(2, VOCAB_SIZE, (B * label_len,))  # avoid blank/unk
        target_lengths = torch.full((B,), label_len, dtype=torch.long)
        input_lengths = torch.full((B,), T, dtype=torch.long)
        return logits, targets, input_lengths, target_lengths

    def test_loss_is_scalar(self, model):
        criterion = build_ctc_loss(blank=BLANK_IDX)
        logits, targets, input_lengths, target_lengths = self._make_valid_batch(model)
        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        assert loss.ndim == 0  # scalar

    def test_loss_is_finite(self, model):
        criterion = build_ctc_loss(blank=BLANK_IDX)
        logits, targets, input_lengths, target_lengths = self._make_valid_batch(model)
        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        assert torch.isfinite(loss), f"CTC loss was not finite: {loss}"

    def test_loss_is_nonnegative(self, model):
        criterion = build_ctc_loss(blank=BLANK_IDX)
        logits, targets, input_lengths, target_lengths = self._make_valid_batch(model)
        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        assert loss.item() >= 0.0

    def test_blank_index_matches_tokenizer(self):
        """Verify the tokenizer's blank_index is 0, matching build_ctc_loss default."""
        tok = CharTokenizer()
        tok.build_vocab(["hello"])
        assert tok.blank_index == BLANK_IDX == 0

    def test_ctc_loss_zero_infinity(self):
        """zero_infinity=True should prevent NaN from short sequences."""
        criterion = build_ctc_loss(blank=BLANK_IDX)
        # Degenerate: T=1, label_len=5 — impossible for CTC, should return 0 not NaN
        logits = torch.randn(1, 1, VOCAB_SIZE)
        targets = torch.tensor([2, 3, 4, 5, 6])
        input_lengths = torch.tensor([1])
        target_lengths = torch.tensor([5])
        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        assert not torch.isnan(loss), "Loss should not be NaN with zero_infinity=True"


# ────────────────────────────────────────────────────────────────────────────
# Gradients
# ────────────────────────────────────────────────────────────────────────────

class TestGradients:

    def test_gradients_flow(self, model):
        """After backward(), all parameters must have gradients."""
        criterion = build_ctc_loss(blank=BLANK_IDX)

        x = make_batch(2, 512)
        logits = model(x)
        T = logits.shape[0]
        B = logits.shape[1]
        label_len = 5

        targets = torch.randint(2, VOCAB_SIZE, (B * label_len,))
        target_lengths = torch.full((B,), label_len, dtype=torch.long)
        input_lengths = torch.full((B,), T, dtype=torch.long)

        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step_changes_weights(self, model):
        """Optimizer step must change at least one parameter."""
        criterion = build_ctc_loss(blank=BLANK_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Snapshot initial weights
        before = {n: p.data.clone() for n, p in model.named_parameters()}

        x = make_batch(2, 512)
        logits = model(x)
        T, B = logits.shape[:2]
        targets = torch.randint(2, VOCAB_SIZE, (B * 5,))
        input_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), 5, dtype=torch.long)

        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = any(
            not torch.equal(before[n], p.data)
            for n, p in model.named_parameters()
        )
        assert changed, "No parameters changed after optimizer step"


# ────────────────────────────────────────────────────────────────────────────
# CPU device
# ────────────────────────────────────────────────────────────────────────────

class TestCPUDevice:

    def test_model_on_cpu(self, model):
        device = torch.device("cpu")
        model = model.to(device)
        x = make_batch(2, 512).to(device)
        out = model(x)
        assert out.device.type == "cpu"

    def test_loss_on_cpu(self, model):
        device = torch.device("cpu")
        model = model.to(device)
        criterion = build_ctc_loss(blank=BLANK_IDX)

        x = make_batch(2, 512).to(device)
        logits = model(x)
        T, B = logits.shape[:2]
        targets = torch.randint(2, VOCAB_SIZE, (B * 5,)).to(device)
        input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
        target_lengths = torch.full((B,), 5, dtype=torch.long).to(device)

        loss = compute_ctc_loss(criterion, logits, targets, input_lengths, target_lengths)
        assert loss.device.type == "cpu"
        assert torch.isfinite(loss)


# ────────────────────────────────────────────────────────────────────────────
# Config factory
# ────────────────────────────────────────────────────────────────────────────

class TestFromConfig:

    def test_from_config(self):
        cfg = {
            "cnn_channels": [16, 32, 64, 128],
            "cnn_kernel_size": 3,
            "cnn_dropout": 0.0,
            "rnn_hidden": 64,
            "rnn_layers": 1,
            "rnn_dropout": 0.0,
        }
        model = CRNN.from_config(vocab_size=40, cfg=cfg)
        assert model.vocab_size == 40
        assert model.rnn_hidden == 64

    def test_from_config_defaults(self):
        model = CRNN.from_config(vocab_size=100, cfg={})
        x = make_batch(1, 512)
        out = model(x)
        assert out.shape[2] == 100
