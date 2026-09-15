"""
Models sub-package.

Public API
----------
CRNN   — CNN-BiLSTM-CTC model for handwritten text recognition
"""

from src.models.crnn import CRNN

__all__ = ["CRNN"]
