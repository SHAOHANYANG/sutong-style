from collections.abc import Collection

import numpy as np
import pytest

import retrieval
from retrieval.bm25 import Bm25Index
from retrieval.dense import DenseIndex
from retrieval.hybrid import FUSION_CONFIGS, HybridRetriever
from retrieval.style_index import StyleIndex
from retrieval.types import Document, Hit
from tests.fakes import FakeEmbedder


def _z(value: float = 0.0) -> np.ndarray:
    vector = np.zeros(20, dtype=np.float64)
    vector[0] = value
    return vector


class CountingText:
    def __init__(self, ids: tuple[str, ...]) -> None:
        self._ids = ids
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    def search(self, query: str, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]:
        self.calls.append((query, k, tuple(exclude_ids)))
        kept = [doc_id for doc_id in self._ids if doc_id not in exclude_ids]
        return [
            Hit(id=doc_id, rank=rank, score=1.0 / rank)
            for rank, doc_id in enumerate(kept[:k], start=1)
        ]


class CountingVector:
    def __init__(self, ids: tuple[str, ...], inner: StyleIndex | None = None) -> None:
        self._ids = ids
        self._inner = inner
        self.queries: list[np.ndarray] = []
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    @property
    def ids(self) -> tuple[str, ...]:
        return self._inner.ids if self._inner is not None else self._ids

    def search(self, query: np.ndarray, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]:
        self.queries.append(np.asarray(query, dtype=np.float64).copy())
        self.calls.append((k, tuple(exclude_ids)))
        if self._inner is not None:
            return self._inner.search(query, k, exclude_ids)
        kept = [doc_id for doc_id in self._ids if doc_id not in exclude_ids]
        return [
            Hit(id=doc_id, rank=rank, score=1.0 / rank)
            for rank, doc_id in enumerate(kept[:k], start=1)
        ]


class FixedEncoder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0

    def zscore(self, text: str) -> np.ndarray:
        self.calls += 1
        return self.vector


class FixedPredictor:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0
        self.seen: list[np.ndarray] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.seen.append(np.asarray(x, dtype=np.float64).copy())
        return self.vector


def test_three_routes_are_called_and_fused() -> None:
    bm25 = CountingText(("a", "b"))
    dense = CountingText(("a", "b"))
    style = CountingVector(("a", "b"))
    encoder = FixedEncoder(_z(-1.0))
    predictor = FixedPredictor(_z(1.0))
    retriever = HybridRetriever(bm25, dense, style, encoder, predictor)
    hits = retriever.search("张三去井边", k=2, exclude_ids=["b"])
    assert bm25.calls == [("张三去井边", 20, ("b",))]
    assert dense.calls == [("张三去井边", 20, ("b",))]
    assert style.calls == [(20, ("b",))]
    assert encoder.calls == 1
    assert predictor.calls == 1
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].sources == {"bm25": 1, "dense": 1, "style": 1}


def test_zero_weight_routes_are_not_called() -> None:
    bm25 = CountingText(("a",))
    dense = CountingText(("a",))
    style = CountingVector(("a",))
    encoder = FixedEncoder(_z())
    predictor = FixedPredictor(_z(1.0))
    content = HybridRetriever(
        bm25, dense, style, encoder, predictor, weights=FUSION_CONFIGS["content"]
    )
    content_hits = content.search("张三去井边", k=1)
    assert style.calls == []
    assert encoder.calls == 0
    assert predictor.calls == 0
    assert len(bm25.calls) == 1
    assert len(dense.calls) == 1
    assert content_hits[0].sources == {"bm25": 1, "dense": 1}

    style_only = HybridRetriever(
        bm25, dense, style, encoder, predictor, weights=FUSION_CONFIGS["style"]
    )
    style_hits = style_only.search("张三去井边", k=1)
    assert len(bm25.calls) == 1
    assert len(dense.calls) == 1
    assert len(style.calls) == 1
    assert encoder.calls == 1
    assert style_hits[0].sources == {"style": 1}


def test_style_route_searches_the_predicted_vector() -> None:
    raw = _z(-4.0)
    predicted = _z(4.0)
    style = CountingVector(
        ("raw-doc", "pred-doc"),
        StyleIndex(["raw-doc", "pred-doc"], np.vstack([raw, predicted])),
    )
    encoder = FixedEncoder(raw)
    predictor = FixedPredictor(predicted)
    retriever = HybridRetriever(
        CountingText(("raw-doc", "pred-doc")),
        CountingText(("raw-doc", "pred-doc")),
        style,
        encoder,
        predictor,
        weights=FUSION_CONFIGS["style"],
    )
    hits = retriever.search("张三去井边", k=1)
    assert hits[0].id == "pred-doc"
    assert style.queries[0] == pytest.approx(predicted)
    assert predictor.seen[0] == pytest.approx(raw)
    assert not np.allclose(style.queries[0], raw)


def test_mismatched_id_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="不一致"):
        HybridRetriever(
            CountingText(("a", "b")),
            CountingText(("a", "c")),
            CountingVector(("a", "b")),
            FixedEncoder(_z()),
            FixedPredictor(_z()),
        )


def test_same_ids_in_different_order_are_accepted() -> None:
    HybridRetriever(
        CountingText(("a", "b")),
        CountingText(("b", "a")),
        CountingVector(("a", "b")),
        FixedEncoder(_z()),
        FixedPredictor(_z()),
    )


def test_bad_depth_rrf_k_and_weights_are_rejected() -> None:
    routes = (
        CountingText(("a",)),
        CountingText(("a",)),
        CountingVector(("a",)),
        FixedEncoder(_z()),
        FixedPredictor(_z()),
    )
    with pytest.raises(ValueError, match="depth"):
        HybridRetriever(*routes, depth=0)
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(*routes, rrf_k=0)
    with pytest.raises(ValueError, match="未知"):
        HybridRetriever(*routes, weights={"other": 1.0})
    with pytest.raises(ValueError, match="有限"):
        HybridRetriever(*routes, weights={"bm25": float("nan")})
    with pytest.raises(ValueError, match="负"):
        HybridRetriever(*routes, weights={"style": -1.0})


def test_package_exports_style_and_hybrid_on_demand() -> None:
    from retrieval import HybridRetriever as ExportedHybrid
    from retrieval import StyleIndex as ExportedStyle
    from retrieval.style_index import StyleIndex as DirectStyle

    assert ExportedStyle is DirectStyle
    assert ExportedHybrid is HybridRetriever
    missing = "not_a_thing"
    with pytest.raises(AttributeError, match=missing):
        getattr(retrieval, missing)


def test_real_indexes_with_the_same_ids_construct() -> None:
    documents = [Document(id="a", text="甲在井边"), Document(id="b", text="乙买了米")]
    HybridRetriever(
        Bm25Index(documents),
        DenseIndex(["b", "a"], np.eye(2, 4), FakeEmbedder(dim=4)),
        StyleIndex(["a", "b"], np.vstack([_z(0.0), _z(1.0)])),
        FixedEncoder(_z()),
        FixedPredictor(_z(1.0)),
    )
