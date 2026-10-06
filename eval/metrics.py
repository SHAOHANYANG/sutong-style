"""Recall, hallucination, and profile gaps. Empty source or output sets follow SPEC 3.2."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _deltas(deltas: NDArray[np.float64]) -> NDArray[np.float64]:
    rows = np.asarray(deltas, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("deltas 必须是非空二维数组")
    if not np.isfinite(rows).all():
        raise ValueError("deltas 含非有限值")
    return rows


def signed_mean_delta(deltas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-dimension mean of d. Opposite errors cancel."""
    return np.asarray(_deltas(deltas).mean(axis=0), dtype=np.float64)


def profile_gap(deltas: NDArray[np.float64]) -> float:
    """Set-level gap: mean of the absolute per-dimension means of d."""
    return float(np.mean(np.abs(signed_mean_delta(deltas))))


def profile_gap_per_case(deltas: NDArray[np.float64]) -> NDArray[np.float64]:
    """One scalar per row: mean of |d_i| across dimensions. Aggregate is their mean."""
    return np.asarray(np.mean(np.abs(_deltas(deltas)), axis=1), dtype=np.float64)


def entity_recall(expected: set[str], actual: set[str]) -> float:
    """Fraction of source entities kept. An empty source set scores 1.0."""
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def numeral_recall(expected: set[str], actual: set[str]) -> float:
    """Fraction of source numeral values kept. An empty source set scores 1.0."""
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def hallucination_rate(expected: set[str], actual: set[str]) -> float:
    """Fraction of output entities absent from the source. An empty output scores 0.0."""
    if not actual:
        return 0.0
    return len(actual - expected) / len(actual)
