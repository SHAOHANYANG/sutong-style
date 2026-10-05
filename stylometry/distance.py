"""Diagonal z-score distance to a reference style. Mahalanobis is intentionally unused."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stylometry.features import FEATURE_NAMES, extract
from stylometry.lexicon import LiteraryLexicon

STD_FLOOR = 1e-6


@dataclass(frozen=True)
class StyleReference:
    """Mean and standard deviation fitted on original texts."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    lexicon: LiteraryLexicon

    @classmethod
    def fit(cls, texts: list[str], lexicon: LiteraryLexicon) -> StyleReference:
        """Fit one reference. Standard deviations are floored at 1e-6."""
        if not texts:
            raise ValueError("风格参照至少需要一条原文")
        vectors = np.vstack([extract(text, lexicon) for text in texts])
        mean = np.asarray(vectors.mean(axis=0), dtype=np.float64)
        std = np.maximum(vectors.std(axis=0, ddof=0), STD_FLOOR).astype(np.float64)
        return cls(mean=mean, std=std, lexicon=lexicon)

    def distance(self, text: str) -> float:
        """Euclidean distance of the z-scored vector, divided by sqrt(20)."""
        vector = extract(text, self.lexicon)
        standardized = (vector - self.mean) / self.std
        return float(np.linalg.norm(standardized) / np.sqrt(len(FEATURE_NAMES)))
