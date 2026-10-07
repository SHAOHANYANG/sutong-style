"""Reciprocal rank fusion over Hit lists. No I/O and no route-specific types."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from retrieval.types import FusedHit, Hit


def fuse(
    routes: Mapping[str, Sequence[Hit]],
    k: int,
    *,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse route rankings. Default weights are 1, which is unweighted RRF.

    A route with weight 0 contributes neither score nor a source entry. Documents
    that appear only on such routes are omitted. An empty route contributes
    nothing. All-empty input returns an empty list.
    """
    if k <= 0:
        raise ValueError("k 必须是正整数")
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须是正整数")
    chosen = _weights(routes, weights)
    for name in sorted(routes):
        _check_hits(name, routes[name])
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    for name in sorted(routes):
        weight = chosen[name]
        if weight == 0.0:
            continue
        for hit in routes[name]:
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (rrf_k + hit.rank)
            sources.setdefault(hit.id, {})[name] = hit.rank
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [
        FusedHit(
            id=doc_id,
            rank=rank,
            score=scores[doc_id],
            sources=dict(sources[doc_id]),
        )
        for rank, doc_id in enumerate(ordered[:k], start=1)
    ]


def _weights(
    routes: Mapping[str, Sequence[Hit]],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    chosen = {name: 1.0 for name in routes}
    if weights is None:
        return chosen
    for name, weight in weights.items():
        if name not in routes:
            raise ValueError(f"权重里有未知路名: {name}")
        value = float(weight)
        if not math.isfinite(value):
            raise ValueError(f"权重不是有限值: {name}")
        if value < 0.0:
            raise ValueError(f"权重为负: {name}")
        chosen[name] = value
    return chosen


def _check_hits(name: str, hits: Sequence[Hit]) -> None:
    seen: set[str] = set()
    for hit in hits:
        if hit.id in seen:
            raise ValueError(f"路 {name} 含重复 id: {hit.id}")
        seen.add(hit.id)
    ranks = sorted(hit.rank for hit in hits)
    if ranks != list(range(1, len(hits) + 1)):
        raise ValueError(f"路 {name} 的 rank 不是从 1 开始的连续整数")
