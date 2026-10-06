"""Brute-force nearest neighbors in 20-dimensional z space. No vector database."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from retrieval.types import Hit
from stylometry.features import FEATURE_NAMES

DIM = len(FEATURE_NAMES)


class StyleIndex:
    """Euclidean index over caller-supplied z vectors. Higher scores are nearer."""

    def __init__(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != len(set(ids)):
            raise ValueError(f"重复的文档 id: {_duplicate_id(ids)}")
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("向量必须是二维矩阵")
        if matrix.shape[1] != DIM:
            raise ValueError("向量必须是 20 维")
        if matrix.shape[0] != len(ids):
            raise ValueError("向量行数与 id 数不一致")
        if len(ids) > 0 and not np.isfinite(matrix).all():
            raise ValueError("向量含非有限值")
        self._ids = list(ids)
        self._matrix = matrix
        self._row = {doc_id: index for index, doc_id in enumerate(self._ids)}

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)

    def vector(self, doc_id: str) -> np.ndarray:
        return np.asarray(self._matrix[self._row[doc_id]], dtype=np.float64)

    def search(self, query: np.ndarray, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]:
        """Return up to `k` hits. `score` is negative Euclidean distance."""
        if k <= 0:
            raise ValueError("k 必须是正整数")
        if not self._ids:
            return []
        point = np.asarray(query, dtype=np.float64)
        if point.shape != (DIM,):
            raise ValueError("查询向量必须是 20 维")
        if not np.isfinite(point).all():
            raise ValueError("查询向量含非有限值")
        excluded = set(exclude_ids)
        distances = np.linalg.norm(self._matrix - point, axis=1)
        ranked = sorted(
            (
                (doc_id, float(-distances[index]))
                for index, doc_id in enumerate(self._ids)
                if doc_id not in excluded
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            Hit(id=doc_id, rank=rank, score=score)
            for rank, (doc_id, score) in enumerate(ranked[:k], start=1)
        ]


def _duplicate_id(ids: Sequence[str]) -> str:
    seen: set[str] = set()
    for doc_id in ids:
        if doc_id in seen:
            return doc_id
        seen.add(doc_id)
    return ""
