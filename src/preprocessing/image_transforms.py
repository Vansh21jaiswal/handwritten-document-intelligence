"""
image_transforms.py
===================
Image preprocessing pipeline for handwritten text recognition.

Accepts a PIL image from the IAM dataset and produces a normalised
PyTorch float tensor ready to pass into a CNN + sequence model.

Pipeline steps
--------------
1. Convert to grayscale  (L mode, 1 channel)
2. Resize to a fixed height while preserving the aspect ratio
   (width scales proportionally — no distortion)
3. Convert to a float32 tensor in [0, 1]
4. Normalise: output = (pixel - mean) / std

The output tensor shape is  (1, H, W)  where H == target_height
and W varies between images. The DataLoader's collate function is
responsible for padding W to a common width within each batch.

Usage
-----
>>> from src.preprocessing.image_transforms import ImagePreprocessor
>>> prep = ImagePreprocessor(target_height=128)
>>> tensor = prep(pil_image)       # shape: (1, 128, W)
>>> tensor = prep.transform(pil_image)   # identical via callable

Config-driven usage
-------------------
>>> import yaml
>>> cfg = yaml.safe_load(open("configs/default.yaml"))["preprocessing"]
>>> prep = ImagePreprocessor.from_config(cfg)
"""

from __future__ import annotations

from typing import Dict, Any

import torch
from PIL import Image


class ImagePreprocessor:
    """Stateless preprocessing pipeline from PIL Image → normalised tensor.

    Parameters
    ----------
    target_height : int
        Output image height in pixels. Width is scaled proportionally.
    mean : float
        Normalisation mean applied to every pixel after scaling to [0, 1].
    std : float
        Normalisation standard deviation.
    """

    def __init__(
        self,
        target_height: int = 128,
        mean: float = 0.5,
        std: float = 0.5,
    ) -> None:
        if target_height <= 0:
            raise ValueError(f"target_height must be positive, got {target_height}")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")

        self.target_height = target_height
        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._process(image)

    def transform(self, image: Image.Image) -> torch.Tensor:
        """Alias for __call__ — useful when passing as a transform argument."""
        return self._process(image)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _process(self, image: Image.Image) -> torch.Tensor:
        """Apply the full preprocessing pipeline.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image in any mode.

        Returns
        -------
        torch.Tensor
            Shape (1, target_height, W), dtype=torch.float32,
            values normalised around mean/std.
        """
        # Step 1 — Grayscale
        image = image.convert("L")

        # Step 2 — Aspect-ratio-preserving resize
        orig_w, orig_h = image.size  # PIL: (width, height)
        scale = self.target_height / orig_h
        new_w = max(1, round(orig_w * scale))
        image = image.resize((new_w, self.target_height), Image.LANCZOS)

        # Step 3 — Tensor: H×W float32 in [0, 1]
        # PIL 'L' mode → numpy would be H×W; we want (1, H, W)
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(image)  # shape: (1, H, W), range [0,1]

        # Step 4 — Normalise
        tensor = TF.normalize(tensor, mean=[self.mean], std=[self.std])

        return tensor  # (1, target_height, W)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImagePreprocessor":
        """Construct from the 'preprocessing' section of default.yaml.

        Parameters
        ----------
        config : dict
            Must contain 'target_height'. Optional keys: 'mean', 'std'.

        Example
        -------
        >>> import yaml
        >>> cfg = yaml.safe_load(open("configs/default.yaml"))
        >>> prep = ImagePreprocessor.from_config(cfg["preprocessing"])
        """
        return cls(
            target_height=config.get("target_height", 128),
            mean=config.get("mean", 0.5),
            std=config.get("std", 0.5),
        )

    def __repr__(self) -> str:
        return (
            f"ImagePreprocessor("
            f"target_height={self.target_height}, "
            f"mean={self.mean}, "
            f"std={self.std})"
        )
