"""Online style score: negative MAE to the StylePredictor target. Higher is better."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from retrieval.style_predictor import StylePredictor
from stylometry.distance import StyleReference


class VectorPredictor(Protocol):
    """Predict a target z vector from a vernacular z vector."""

    def predict(self, x: np.ndarray) -> np.ndarray: ...


class PredictorScorer:
    """Candidate B from the scorer audit. Only breaks ties among equal violation counts."""

    def __init__(self, reference: StyleReference, predictor: VectorPredictor) -> None:
        self.reference = reference
        self.predictor = predictor

    def score(self, vernacular: str, output: str) -> float:
        target = np.asarray(
            self.predictor.predict(self.reference.zscore(vernacular)),
            dtype=np.float64,
        )
        if target.ndim == 2:
            target = target[0]
        observed = self.reference.zscore(output)
        mae = float(np.mean(np.abs(observed - target)))
        return -mae


def fit_train_predictor(
    reference: StyleReference,
    vernaculars: list[str],
    originals: list[str],
    works: list[str],
) -> StylePredictor:
    """Fit on the caller's train rows. Kept here so the scorer module stays the assembly point."""
    from retrieval.style_predictor import fit_predictor

    x = np.vstack([reference.zscore(text) for text in vernaculars])
    y = np.vstack([reference.zscore(text) for text in originals])
    return fit_predictor(x, y, works)
