"""
Evaluation sub-package.

Public API
----------
compute_cer       — aggregate Character Error Rate
compute_wer       — aggregate Word Error Rate
per_sample_metrics— CER/WER per (reference, hypothesis) pair
evaluate          — full evaluation returning EvaluationResult
EvaluationResult  — dataclass: cer, wer, n_samples, samples
SampleResult      — dataclass: reference, hypothesis, cer, wer
"""

from src.evaluation.metrics import (
    compute_cer,
    compute_wer,
    per_sample_metrics,
    evaluate,
    EvaluationResult,
    SampleResult,
)

__all__ = [
    "compute_cer",
    "compute_wer",
    "per_sample_metrics",
    "evaluate",
    "EvaluationResult",
    "SampleResult",
]
