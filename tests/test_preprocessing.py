"""
test_preprocessing.py
=====================
Unit tests for the image preprocessing pipeline.

Uses small synthetic PIL images — no IAM dataset download required.
"""

import pytest
import torch
from PIL import Image

from src.preprocessing import ImagePreprocessor


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def preprocessor():
    return ImagePreprocessor(target_height=128, mean=0.5, std=0.5)


def make_rgb_image(width: int = 300, height: int = 64) -> Image.Image:
    """Synthetic RGB image with random-ish pixels."""
    import random
    pixels = [random.randint(0, 255) for _ in range(width * height * 3)]
    img = Image.new("RGB", (width, height))
    img.putdata(list(zip(pixels[::3], pixels[1::3], pixels[2::3])))
    return img


def make_grayscale_image(width: int = 256, height: int = 32) -> Image.Image:
    img = Image.new("L", (width, height), color=128)
    return img


# ────────────────────────────────────────────────────────────────────────────
# Output shape tests
# ────────────────────────────────────────────────────────────────────────────

class TestOutputShape:

    def test_output_is_tensor(self, preprocessor):
        img = make_rgb_image(200, 50)
        result = preprocessor(img)
        assert isinstance(result, torch.Tensor)

    def test_output_channels(self, preprocessor):
        """Output must be single-channel (grayscale)."""
        img = make_rgb_image(300, 80)
        result = preprocessor(img)
        assert result.shape[0] == 1, f"Expected 1 channel, got {result.shape[0]}"

    def test_output_height_fixed(self, preprocessor):
        """Height must always equal target_height regardless of input."""
        for h in [32, 64, 128, 256]:
            img = make_rgb_image(200, h)
            result = preprocessor(img)
            assert result.shape[1] == 128, (
                f"Input height {h} → output height {result.shape[1]}, expected 128"
            )

    def test_aspect_ratio_preserved(self, preprocessor):
        """Width should scale proportionally to keep aspect ratio."""
        # Input: 256×64 → scale = 128/64 = 2.0 → expected W = 512
        img = make_rgb_image(width=256, height=64)
        result = preprocessor(img)
        assert result.shape[2] == 512, (
            f"Expected width 512, got {result.shape[2]}"
        )

    def test_wide_image_preserves_width_scale(self, preprocessor):
        # Input: 1000×200 → scale = 128/200 = 0.64 → expected W = 640
        img = make_rgb_image(width=1000, height=200)
        result = preprocessor(img)
        assert result.shape[2] == 640, (
            f"Expected width 640, got {result.shape[2]}"
        )

    def test_grayscale_input_accepted(self, preprocessor):
        img = make_grayscale_image(200, 50)
        result = preprocessor(img)
        assert result.shape[0] == 1
        assert result.shape[1] == 128

    def test_different_target_heights(self):
        for target_h in [32, 64, 96]:
            prep = ImagePreprocessor(target_height=target_h)
            img = make_rgb_image(200, 50)
            result = prep(img)
            assert result.shape[1] == target_h


# ────────────────────────────────────────────────────────────────────────────
# Value range tests
# ────────────────────────────────────────────────────────────────────────────

class TestOutputValues:

    def test_dtype_float32(self, preprocessor):
        img = make_rgb_image(200, 64)
        result = preprocessor(img)
        assert result.dtype == torch.float32

    def test_normalised_range(self, preprocessor):
        """With mean=0.5, std=0.5, pixels in [0,1] map to [-1, 1]."""
        img = make_rgb_image(200, 64)
        result = preprocessor(img)
        # Should be broadly in [-1, 1] for a mean=0.5, std=0.5 normalisation
        assert result.min() >= -1.1, f"Min value {result.min()} below expected range"
        assert result.max() <= 1.1, f"Max value {result.max()} above expected range"

    def test_pure_white_image(self):
        """Pure white (255) normalised with mean=0.5, std=0.5 → 1.0."""
        prep = ImagePreprocessor(target_height=32, mean=0.5, std=0.5)
        img = Image.new("L", (100, 32), color=255)
        result = prep(img)
        torch.testing.assert_close(result, torch.ones_like(result))

    def test_pure_black_image(self):
        """Pure black (0) normalised with mean=0.5, std=0.5 → -1.0."""
        prep = ImagePreprocessor(target_height=32, mean=0.5, std=0.5)
        img = Image.new("L", (100, 32), color=0)
        result = prep(img)
        torch.testing.assert_close(result, torch.full_like(result, -1.0))


# ────────────────────────────────────────────────────────────────────────────
# Config factory
# ────────────────────────────────────────────────────────────────────────────

class TestFromConfig:

    def test_from_config_defaults(self):
        cfg = {"target_height": 64, "mean": 0.4, "std": 0.3}
        prep = ImagePreprocessor.from_config(cfg)
        assert prep.target_height == 64
        assert prep.mean == 0.4
        assert prep.std == 0.3

    def test_from_config_partial(self):
        """Missing keys fall back to defaults."""
        prep = ImagePreprocessor.from_config({"target_height": 128})
        assert prep.mean == 0.5
        assert prep.std == 0.5


# ────────────────────────────────────────────────────────────────────────────
# Edge cases
# ────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_invalid_target_height(self):
        with pytest.raises(ValueError):
            ImagePreprocessor(target_height=0)

    def test_invalid_std(self):
        with pytest.raises(ValueError):
            ImagePreprocessor(target_height=128, std=0)

    def test_transform_alias(self, preprocessor):
        img = make_rgb_image(200, 64)
        r1 = preprocessor(img)
        r2 = preprocessor.transform(img)
        torch.testing.assert_close(r1, r2)
