"""
Inference sub-package.

Public API
----------
GreedyDecoder  — CTC greedy decoder: argmax → collapse → remove blank → text
"""

from src.inference.greedy_decoder import GreedyDecoder

__all__ = ["GreedyDecoder"]
