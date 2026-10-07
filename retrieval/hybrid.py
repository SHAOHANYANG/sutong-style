"""Assemble the three routes. No corpus I/O.

The objects passed in are a Bm25Index, a DenseIndex, a StyleIndex, a StyleReference,
and a StylePredictor. The constructor types are protocols so tests can count calls.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Protocol

import numpy as np

from retrieval.fusion import fuse
from retrieval.types import FusedHit, Hit

ROUTE_BM25 = "bm25"
ROUTE_DENSE = "dense"
ROUTE_STYLE = "style"
ROUTE_NAMES = (ROUTE_BM25, ROUTE_DENSE, ROUTE_STYLE)
DEFAULT_DEPTH = 20
DEFAULT_RRF_K = 60

# Pre-registered in SPEC 4.6. Names are labels, not a ranking of quality.
FUSION_CONFIGS: dict[str, dict[str, float]] = {
    "equal": {ROUTE_BM25: 1.0, ROUTE_DENSE: 1.0, ROUTE_STYLE: 1.0},
    "balanced": {ROUTE_BM25: 0.5, ROUTE_DENSE: 0.5, ROUTE_STYLE: 1.0},
    "content": {ROUTE_BM25: 1.0, ROUTE_DENSE: 1.0, ROUTE_STYLE: 0.0},
    "style": {ROUTE_BM25: 0.0, ROUTE_DENSE: 0.0, ROUTE_STYLE: 1.0},
}


class TextRoute(Protocol):
    """BM25 or dense: search by the query text."""

    @property
    def ids(self) -> tuple[str, ...]: ...

    def search(self, query: str, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]: ...


class VectorRoute(Protocol):
    """Style index: search by a 20-dimensional vector."""

    @property
    def ids(self) -> tuple[str, ...]: ...

    def search(self, query: np.ndarray, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]: ...


class StyleEncoder(Protocol):
    def zscore(self, text: str) -> np.ndarray: ...


class VectorPredictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class HybridRetriever:
    """BM25 and dense take the text. Style takes the predicted z vector."""

    def __init__(
        self,
        bm25: TextRoute,
        dense: TextRoute,
        style: VectorRoute,
        reference: StyleEncoder,
        predictor: VectorPredictor,
        *,
        depth: int = DEFAULT_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth 必须是正整数")
        if rrf_k <= 0:
            raise ValueError("rrf_k 必须是正整数")
        _require_same_ids(bm25.ids, dense.ids, style.ids)
        self._bm25 = bm25
        self._dense = dense
        self._style = style
        self._reference = reference
        self._predictor = predictor
        self._depth = depth
        self._rrf_k = rrf_k
        self._weights = _resolve_weights(weights)

    def search(self, query: str, k: int, exclude_ids: Collection[str] = ()) -> list[FusedHit]:
        """Fuse the top `depth` hits from each route that has a positive weight."""
        routes: dict[str, list[Hit]] = {
            ROUTE_BM25: [],
            ROUTE_DENSE: [],
            ROUTE_STYLE: [],
        }
        if self._weights[ROUTE_BM25] != 0.0:
            routes[ROUTE_BM25] = self._bm25.search(query, self._depth, exclude_ids)
        if self._weights[ROUTE_DENSE] != 0.0:
            routes[ROUTE_DENSE] = self._dense.search(query, self._depth, exclude_ids)
        if self._weights[ROUTE_STYLE] != 0.0:
            raw = np.asarray(self._reference.zscore(query), dtype=np.float64)
            predicted = np.asarray(self._predictor.predict(raw), dtype=np.float64)
            routes[ROUTE_STYLE] = self._style.search(predicted, self._depth, exclude_ids)
        return fuse(routes, k, rrf_k=self._rrf_k, weights=self._weights)


def _require_same_ids(
    bm25: tuple[str, ...], dense: tuple[str, ...], style: tuple[str, ...]
) -> None:
    if set(bm25) != set(dense) or set(bm25) != set(style):
        raise ValueError("三个索引的 id 集合不一致")


def _resolve_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    chosen = {name: 1.0 for name in ROUTE_NAMES}
    if weights is None:
        return chosen
    for name, weight in weights.items():
        if name not in chosen:
            raise ValueError(f"权重里有未知路名: {name}")
        value = float(weight)
        if not math.isfinite(value):
            raise ValueError(f"权重不是有限值: {name}")
        if value < 0.0:
            raise ValueError(f"权重为负: {name}")
        chosen[name] = value
    return chosen
