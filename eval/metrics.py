"""Recall and hallucination. Empty source or output sets follow SPEC 3.2."""

from __future__ import annotations


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
