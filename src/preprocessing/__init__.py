"""
Preprocessing sub-package.

Public API
----------
ImagePreprocessor   — PIL Image → normalised (1, H, W) torch.Tensor
"""

from src.preprocessing.image_transforms import ImagePreprocessor

__all__ = ["ImagePreprocessor"]
