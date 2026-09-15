"""
test_dataset_pipeline.py
========================
Unit tests for IAMTorchDataset and the collate function.

Uses small synthetic examples — NO IAM dataset download required.
"""

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.preprocessing import ImagePreprocessor
from src.dataset.tokenizer import CharTokenizer
from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn


# ────────────────────────────────────────────────────────────────────────────
# Synthetic helpers
# ────────────────────────────────────────────────────────────────────────────

def make_fake_hf_split(texts, widths=None, height=128):
    """Create a list-of-dicts that mimics a HF Dataset split."""
    if widths is None:
        widths = [200 + i * 50 for i in range(len(texts))]

    class FakeSplit:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            return self._rows[idx]

    rows = []
    for text, w in zip(texts, widths):
        img = Image.new("L", (w, height), color=200)
        rows.append({"image": img, "text": text})
    return FakeSplit(rows)


TRAIN_TEXTS = ["hello world", "foo bar baz", "the quick brown fox jumps"]
FAKE_SPLIT = make_fake_hf_split(TRAIN_TEXTS, widths=[200, 300, 400])


@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.build_vocab(TRAIN_TEXTS)
    return tok


@pytest.fixture
def preprocessor():
    return ImagePreprocessor(target_height=128)


@pytest.fixture
def dataset(tokenizer, preprocessor):
    return IAMTorchDataset(FAKE_SPLIT, preprocessor, tokenizer)


# ────────────────────────────────────────────────────────────────────────────
# IAMTorchDataset
# ────────────────────────────────────────────────────────────────────────────

class TestIAMTorchDataset:

    def test_len(self, dataset):
        assert len(dataset) == len(TRAIN_TEXTS)

    def test_getitem_returns_dict_keys(self, dataset):
        item = dataset[0]
        assert set(item.keys()) == {"image", "target", "target_length", "text", "image_width"}

    def test_image_tensor_shape(self, dataset):
        item = dataset[0]
        t = item["image"]
        assert t.ndim == 3
        assert t.shape[0] == 1       # single channel
        assert t.shape[1] == 128     # fixed height

    def test_image_dtype(self, dataset):
        assert dataset[0]["image"].dtype == torch.float32

    def test_target_is_tensor(self, dataset):
        assert isinstance(dataset[0]["target"], torch.Tensor)

    def test_target_dtype(self, dataset):
        assert dataset[0]["target"].dtype == torch.long

    def test_target_length_matches_text(self, dataset):
        item = dataset[0]
        assert item["target_length"] == len(item["text"])

    def test_target_length_matches_target_tensor(self, dataset):
        item = dataset[0]
        assert item["target_length"] == len(item["target"])

    def test_text_preserved(self, dataset):
        item = dataset[0]
        assert item["text"] == TRAIN_TEXTS[0]

    def test_image_width_positive(self, dataset):
        item = dataset[0]
        assert item["image_width"] > 0

    def test_unbuilt_tokenizer_raises(self, preprocessor):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="build_vocab"):
            IAMTorchDataset(FAKE_SPLIT, preprocessor, tok)

    def test_vocab_size_property(self, dataset):
        assert dataset.vocab_size == dataset.tokenizer.vocab_size


# ────────────────────────────────────────────────────────────────────────────
# Collate function
# ────────────────────────────────────────────────────────────────────────────

class TestCollateFunction:

    @pytest.fixture
    def batch(self, dataset):
        collate = build_collate_fn(pad_value=0.0)
        items = [dataset[i] for i in range(len(dataset))]
        return collate(items)

    def test_batch_keys(self, batch):
        assert set(batch.keys()) == {
            "images", "targets", "target_lengths", "image_widths", "texts"
        }

    def test_images_batch_dim(self, batch):
        assert batch["images"].shape[0] == len(TRAIN_TEXTS)

    def test_images_height_fixed(self, batch):
        assert batch["images"].shape[2] == 128

    def test_images_width_is_max(self, batch):
        # Widths scale proportionally. Input widths: 200, 300, 400 at height 128.
        # All inputs already at height 128, so scale=1; widths stay 200, 300, 400.
        max_w = batch["image_widths"].max().item()
        assert batch["images"].shape[3] == max_w

    def test_images_channels(self, batch):
        assert batch["images"].shape[1] == 1

    def test_targets_flat_1d(self, batch):
        assert batch["targets"].ndim == 1

    def test_targets_length_sum(self, batch):
        expected_total = sum(len(t) for t in TRAIN_TEXTS)
        assert len(batch["targets"]) == expected_total

    def test_target_lengths_shape(self, batch):
        assert batch["target_lengths"].shape == (len(TRAIN_TEXTS),)

    def test_target_lengths_values(self, batch):
        for i, text in enumerate(TRAIN_TEXTS):
            assert batch["target_lengths"][i].item() == len(text)

    def test_image_widths_shape(self, batch):
        assert batch["image_widths"].shape == (len(TRAIN_TEXTS),)

    def test_image_widths_ascending(self, batch):
        # Input widths 200 < 300 < 400 → processed widths stay ordered
        widths = batch["image_widths"].tolist()
        assert widths == sorted(widths)

    def test_texts_preserved(self, batch):
        assert batch["texts"] == TRAIN_TEXTS

    def test_padding_fills_pad_value(self, batch):
        """Positions beyond the original width of the first (narrowest) image
        should be filled with the pad value."""
        max_w = batch["images"].shape[3]
        first_w = batch["image_widths"][0].item()
        if first_w < max_w:
            # The padded region of sample 0 should equal pad_value=0.0
            padded_region = batch["images"][0, 0, :, first_w:]
            assert torch.all(padded_region == 0.0), (
                "Padded region should be 0.0 (pad_value)"
            )


# ────────────────────────────────────────────────────────────────────────────
# DataLoader integration
# ────────────────────────────────────────────────────────────────────────────

class TestDataLoaderIntegration:

    def test_dataloader_one_batch(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=build_collate_fn(),
            shuffle=False,
        )
        batch = next(iter(loader))
        assert batch["images"].shape[0] == 2

    def test_dataloader_iterates_fully(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=build_collate_fn(),
        )
        total = sum(b["images"].shape[0] for b in loader)
        assert total == len(dataset)
