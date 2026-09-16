"""
metrics.py
==========
CER (Character Error Rate) and WER (Word Error Rate) evaluation utilities.

Uses the `jiwer` library for efficient edit-distance computation.

Definitions
-----------
CER = edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)
WER = edit_distance(reference_words, hypothesis_words) / len(reference_words)

Both metrics range from 0 (perfect) to ≥1 (poor). Values >1 occur when the
number of insertion errors exceeds the reference length.

Usage
-----
>>> from src.evaluation.metrics import compute_cer, compute_wer, evaluate
>>> cer = compute_cer(["hello world"], ["helo wrld"])
>>> wer = compute_wer(["hello world"], ["helo wrld"])
>>> results = evaluate(references, hypotheses)   # dict with cer, wer, samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import jiwer


# ── jiwer transform pipelines ─────────────────────────────────────────────────
# Character-level: split every character into a "word" for jiwer
_CHAR_TRANSFORM = jiwer.Compose([
    jiwer.ReduceToListOfListOfChars(),
])

# Word-level: standard whitespace tokenisation
_WORD_TRANSFORM = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


# ────────────────────────────────────────────────────────────────────────────
# Per-metric functions
# ────────────────────────────────────────────────────────────────────────────

def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """Compute aggregate Character Error Rate over a list of string pairs.

    Parameters
    ----------
    references : List[str]
        Ground-truth transcriptions.
    hypotheses : List[str]
        Model predictions.

    Returns
    -------
    float
        CER in [0, ∞). Values > 1 indicate more errors than reference chars.
    """
    if not references:
        return 0.0
    return jiwer.cer(references, hypotheses)


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Compute aggregate Word Error Rate over a list of string pairs.

    Parameters
    ----------
    references : List[str]
        Ground-truth transcriptions.
    hypotheses : List[str]
        Model predictions.

    Returns
    -------
    float
        WER in [0, ∞).
    """
    if not references:
        return 0.0
    return jiwer.wer(references, hypotheses)


# ────────────────────────────────────────────────────────────────────────────
# Per-sample analysis
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Evaluation result for a single (reference, hypothesis) pair."""
    reference: str
    hypothesis: str
    cer: float
    wer: float


def per_sample_metrics(
    references: List[str],
    hypotheses: List[str],
) -> List[SampleResult]:
    """Compute CER and WER for every individual (reference, hypothesis) pair.

    Parameters
    ----------
    references : List[str]
    hypotheses : List[str]

    Returns
    -------
    List[SampleResult]
        One SampleResult per pair.
    """
    results = []
    for ref, hyp in zip(references, hypotheses):
        sample_cer = jiwer.cer([ref], [hyp])
        sample_wer = jiwer.wer([ref], [hyp])
        results.append(SampleResult(
            reference=ref,
            hypothesis=hyp,
            cer=sample_cer,
            wer=sample_wer,
        ))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Aggregate evaluation
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Aggregate evaluation metrics over a full split."""
    cer: float
    wer: float
    n_samples: int
    samples: List[SampleResult] = field(default_factory=list)


def evaluate(
    references: List[str],
    hypotheses: List[str],
    include_samples: bool = False,
) -> EvaluationResult:
    """Compute aggregate CER and WER, optionally with per-sample breakdown.

    Parameters
    ----------
    references : List[str]
        Ground-truth transcriptions.
    hypotheses : List[str]
        Model predictions (same order as references).
    include_samples : bool
        If True, compute per-sample CER/WER (slower).

    Returns
    -------
    EvaluationResult
    """
    cer = compute_cer(references, hypotheses)
    wer = compute_wer(references, hypotheses)
    samples = per_sample_metrics(references, hypotheses) if include_samples else []
    return EvaluationResult(cer=cer, wer=wer, n_samples=len(references), samples=samples)
