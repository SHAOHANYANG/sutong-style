import ast
import json
from pathlib import Path

import numpy as np
import pytest

from retrieval import DenseCacheError, DenseIndex, Document, load_dense_cache, save_dense_cache
from scripts.build_dense_index import train_documents
from scripts.chunk_corpus import CorpusChunk
from scripts.split_corpus import CorpusSplit
from tests.fakes import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]


def _embedder() -> FakeEmbedder:
    return FakeEmbedder(
        dim=2,
        vectors={
            "甲": np.array([3.0, 0.0]),
            "乙": np.array([0.0, 4.0]),
            "问": np.array([2.0, 0.0]),
            "反": np.array([-5.0, 0.0]),
            "零": np.zeros(2),
        },
    )


def _documents() -> list[Document]:
    return [Document(id="b", text="乙"), Document(id="a", text="甲")]


def test_index_normalizes_and_ranks_by_cosine() -> None:
    hits = DenseIndex.from_documents(_documents(), _embedder()).search("问", k=2)
    assert [(hit.id, hit.rank) for hit in hits] == [("a", 1), ("b", 2)]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)


def test_negative_cosine_is_kept() -> None:
    hits = DenseIndex.from_documents([Document(id="a", text="甲")], _embedder()).search("反", k=1)
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].score == pytest.approx(-1.0)


def test_ties_break_by_id_ascending() -> None:
    embedder = FakeEmbedder(
        dim=2,
        vectors={
            "甲": np.array([1.0, 0.0]),
            "乙": np.array([2.0, 0.0]),
            "问": np.array([1.0, 0.0]),
        },
    )
    hits = DenseIndex.from_documents(_documents(), embedder).search("问", k=5)
    assert [(hit.id, hit.rank) for hit in hits] == [("a", 1), ("b", 2)]
    assert hits[0].score == hits[1].score


def test_exclude_ids_renumbers_ranks_from_one() -> None:
    hits = DenseIndex.from_documents(_documents(), _embedder()).search(
        "问", k=5, exclude_ids=["a", "missing"]
    )
    assert [(hit.id, hit.rank) for hit in hits] == [("b", 1)]


def test_k_boundaries() -> None:
    index = DenseIndex.from_documents(_documents(), _embedder())
    assert [hit.rank for hit in index.search("问", k=10)] == [1, 2]
    with pytest.raises(ValueError, match="k 必须是正整数"):
        index.search("问", k=0)
    with pytest.raises(ValueError, match="k 必须是正整数"):
        index.search("问", k=-1)


def test_empty_index_does_not_call_the_embedder() -> None:
    embedder = FakeEmbedder()
    index = DenseIndex.from_documents([], embedder)
    assert index.search("问", k=3) == []
    assert embedder.calls == []


def test_zero_query_returns_no_hits() -> None:
    index = DenseIndex.from_documents([Document(id="a", text="甲")], _embedder())
    assert index.search("零", k=1) == []


def test_duplicate_id_is_rejected() -> None:
    documents = [Document(id="a", text="甲"), Document(id="a", text="乙")]
    with pytest.raises(ValueError, match="重复的文档 id"):
        DenseIndex.from_documents(documents, _embedder())


def test_bad_matrices_are_rejected() -> None:
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="二维"):
        DenseIndex(["a"], np.array([1.0, 0.0]), embedder)
    with pytest.raises(ValueError, match="行数"):
        DenseIndex(["a"], np.ones((2, 2)), embedder)
    with pytest.raises(ValueError, match="非有限"):
        DenseIndex(["a"], np.array([[np.nan, 1.0]]), embedder)
    with pytest.raises(ValueError, match="零向量"):
        DenseIndex(["a"], np.zeros((1, 2)), embedder)


class _FixedEmbedder:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.vector.shape[0]), dtype=np.float64)
        return np.vstack([self.vector for _ in texts])


def test_query_shape_and_non_finite_values_are_rejected() -> None:
    matrix = np.array([[1.0, 0.0]])
    wide = DenseIndex(["a"], matrix, _FixedEmbedder(np.array([1.0, 0.0, 0.0])))
    with pytest.raises(ValueError, match="形状"):
        wide.search("问", k=1)
    broken = DenseIndex(["a"], matrix, _FixedEmbedder(np.array([np.nan, 1.0])))
    with pytest.raises(ValueError, match="非有限"):
        broken.search("问", k=1)


def test_cache_roundtrip_and_mismatches(tmp_path: Path) -> None:
    documents = [Document(id="a", text="独有句子甲"), Document(id="b", text="独有句子乙")]
    raw = np.array([[3.0, 0.0], [0.0, 4.0]])
    save_dense_cache(
        documents,
        raw,
        model_id="BAAI/bge-m3",
        revision="abc123",
        max_length=512,
        directory=tmp_path,
    )
    meta_text = (tmp_path / "embeddings.meta.json").read_text(encoding="utf-8")
    assert "独有句子甲" not in meta_text
    assert "独有句子乙" not in meta_text
    loaded = load_dense_cache(
        documents,
        model_id="BAAI/bge-m3",
        revision="abc123",
        max_length=512,
        directory=tmp_path,
    )
    assert np.allclose(loaded, raw)

    swapped = [documents[1], documents[0]]
    with pytest.raises(DenseCacheError, match="id 顺序") as caught:
        load_dense_cache(
            swapped,
            model_id="BAAI/bge-m3",
            revision="abc123",
            max_length=512,
            directory=tmp_path,
        )
    assert "独有句子甲" not in str(caught.value)

    changed = [documents[0], Document(id="b", text="改过的句子")]
    with pytest.raises(DenseCacheError, match="sha256"):
        load_dense_cache(
            changed,
            model_id="BAAI/bge-m3",
            revision="abc123",
            max_length=512,
            directory=tmp_path,
        )
    with pytest.raises(DenseCacheError, match="模型标识"):
        load_dense_cache(
            documents,
            model_id="other",
            revision="abc123",
            max_length=512,
            directory=tmp_path,
        )
    with pytest.raises(DenseCacheError, match="revision"):
        load_dense_cache(
            documents,
            model_id="BAAI/bge-m3",
            revision="def456",
            max_length=512,
            directory=tmp_path,
        )
    with pytest.raises(DenseCacheError, match="max_length"):
        load_dense_cache(
            documents,
            model_id="BAAI/bge-m3",
            revision="abc123",
            max_length=256,
            directory=tmp_path,
        )

    payload = json.loads(meta_text)
    payload["dim"] = 4
    (tmp_path / "embeddings.meta.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DenseCacheError, match="形状"):
        load_dense_cache(
            documents,
            model_id="BAAI/bge-m3",
            revision="abc123",
            max_length=512,
            directory=tmp_path,
        )


def test_incomplete_cache_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DenseCacheError, match="不完整"):
        load_dense_cache(
            [Document(id="a", text="甲")],
            model_id="BAAI/bge-m3",
            revision="abc123",
            max_length=512,
            directory=tmp_path,
        )


def test_fake_embedder_is_deterministic() -> None:
    first = FakeEmbedder(dim=4).encode(["甲", "乙"])
    second = FakeEmbedder(dim=4).encode(["甲", "乙"])
    assert first.shape == (2, 4)
    assert np.allclose(first, second)
    assert FakeEmbedder(dim=4).encode([]).shape == (0, 4)


def test_train_documents_follow_split_order_and_drop_eval() -> None:
    chunks = [
        CorpusChunk(id="b", work="自编", idx=2, original="乙"),
        CorpusChunk(id="e", work="自编", idx=3, original="评"),
        CorpusChunk(id="a", work="自编", idx=1, original="甲"),
    ]
    split = CorpusSplit(seed=42, eval_ratio=0.08, train=["a", "b"], eval=["e"])
    documents = train_documents(chunks, split)
    assert [(document.id, document.text) for document in documents] == [("a", "甲"), ("b", "乙")]


def test_train_documents_reject_missing_unknown_and_duplicate_ids() -> None:
    split = CorpusSplit(seed=42, eval_ratio=0.08, train=["a"], eval=[])
    with pytest.raises(ValueError, match="缺失"):
        train_documents([], split)
    with pytest.raises(ValueError, match="不在 split"):
        train_documents([CorpusChunk(id="z", work="自编", idx=1, original="甲")], split)
    duplicated = CorpusSplit(seed=42, eval_ratio=0.08, train=["a", "a"], eval=[])
    with pytest.raises(ValueError, match="重复"):
        train_documents([CorpusChunk(id="a", work="自编", idx=1, original="甲")], duplicated)


def test_dense_modules_do_not_import_the_model_stack_at_import_time() -> None:
    dense_imports = _top_level_imports(ROOT / "retrieval" / "dense.py")
    script_imports = _top_level_imports(ROOT / "scripts" / "build_dense_index.py")
    embedder_imports = _top_level_imports(ROOT / "infra" / "bge_embedder.py")
    forbidden = {"torch", "transformers", "flagembedding", "psycopg"}
    assert dense_imports.isdisjoint(forbidden)
    assert script_imports.isdisjoint(forbidden | {"infra"})
    assert embedder_imports.isdisjoint({"flagembedding", "psycopg"})
    embedder_source = (ROOT / "infra" / "bge_embedder.py").read_text(encoding="utf-8")
    assert "last_hidden_state" in embedder_source


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module.split(".", maxsplit=1)[0])
    return names
