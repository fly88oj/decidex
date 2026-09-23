"""Probability math with the exact semantics the official Jev API documents.

- ``confidence = clamp((K * p_max - 1) / (K - 1), 0, 1)``

  Published in the interactive demo source of the official Confidence docs page
  (``choiceConfidence`` in docs.typesafe.ai/confidence). Verified against both
  worked examples in the official API reference:
  p_max=0.88, K=3 -> 0.82 (docs show 0.81); p_max=0.95, K=3 -> 0.925 (0.92).

- ``score = sum(i * p_i)`` over ordered levels

  The official API reference states the score is "the probability-weighted
  answer across the levels; can land between levels", and its example
  {0: 0.0, 1: 0.95, 2: 0.05} -> 1.05 matches this formula exactly.
"""

from __future__ import annotations

import math

# Probabilities are rounded to this many decimals in responses; rounding drift
# is corrected onto the argmax so the distribution still sums to exactly 1.0.
PROB_DECIMALS = 4


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax with temperature. Temperature must be > 0."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def confidence(probs: list[float]) -> float:
    """Official confidence statistic from a probability distribution."""
    if not probs:
        raise ValueError("probabilities must be non-empty")
    k = len(probs)
    if k == 1:
        return 1.0
    peak = max(probs)
    return max(0.0, min(1.0, (k * peak - 1.0) / (k - 1.0)))


def weighted_score(probs: list[float]) -> float:
    """Probability-weighted level index: sum(i * p_i). Can land between levels."""
    return float(sum(i * p for i, p in enumerate(probs)))


def round_distribution(probs: list[float], decimals: int = PROB_DECIMALS) -> list[float]:
    """Round probabilities so they still sum to exactly 1.0.

    Drift from rounding is absorbed by the largest entry (argmax), which keeps
    the reported winner stable and mirrors the official examples where the
    probabilities visibly sum to 1.
    """
    if not probs:
        return []
    rounded = [round(p, decimals) for p in probs]
    drift = round(1.0 - sum(rounded), decimals)
    if drift:
        top = max(range(len(rounded)), key=lambda i: rounded[i])
        rounded[top] = round(rounded[top] + drift, decimals)
    return rounded
