"""Dense retrieval over caller-supplied vectors. Cosine is computed in memory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from retrieval.types import Document, Embedder, Hit

EMBEDDINGS_NAME = "embeddings.npy"
META_NAME = "embeddings.meta.json"
DEFAULT_CACHE_DIR = Path(__file__).parent / "data"


class DenseCacheError(ValueError):
    """The on-disk matrix does not match the documents or the model that should have built it."""


class EmbeddingCacheMeta(BaseModel):
    """Labels for an unlabeled `.npy`. Text stays out; only ids and hashes are stored."""

    ids: list[str]
    text_sha256: list[str]
    model_id: str
    revision: str
    dim: int
    max_length: int
    created_at: str


class DenseIndex:
    """In-memory cosine index. Rows are L2-normalized here, whatever the embedder did."""

    def __init__(self, ids: Sequence[str], embeddings: np.ndarray, embedder: Embedder) -> None:
        if len(ids) != len(set(ids)):
            raise ValueError(f"重复的文档 id: {_duplicate_id(ids)}")
        matrix = _as_float_matrix(embeddings)
        if matrix.ndim != 2:
            raise ValueError("向量必须是二维矩阵")
        if matrix.shape[0] != len(ids):
            raise ValueError("向量行数与 id 数不一致")
        self._embedder = embedder
        self._ids = list(ids)
        if len(ids) == 0:
            self._matrix = matrix
            return
        if not np.isfinite(matrix).all():
            raise ValueError("向量含非有限值")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("索引含零向量")
        self._matrix = matrix / norms[:, None]

    @classmethod
    def from_documents(cls, documents: Sequence[Document], embedder: Embedder) -> DenseIndex:
        """Encode originals on the spot. Empty input does not call the embedder."""
        if not documents:
            return cls([], np.zeros((0, 1), dtype=np.float64), embedder)
        ids = [document.id for document in documents]
        vectors = embedder.encode([document.text for document in documents])
        return cls(ids, vectors, embedder)

    def search(self, query: str, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]:
        """Return up to `k` hits. Cosine has no zero point, so low scores are kept."""
        if k <= 0:
            raise ValueError("k 必须是正整数")
        if not self._ids:
            return []
        raw = _as_float_matrix(self._embedder.encode([query]))
        if raw.shape != (1, self._matrix.shape[1]):
            raise ValueError("查询向量形状与索引不一致")
        if not np.isfinite(raw).all():
            raise ValueError("查询向量含非有限值")
        norm = float(np.linalg.norm(raw[0]))
        if norm == 0.0:
            return []
        scores = self._matrix @ (raw[0] / norm)
        excluded = set(exclude_ids)
        ranked = sorted(
            (
                (doc_id, float(scores[index]))
                for index, doc_id in enumerate(self._ids)
                if doc_id not in excluded
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            Hit(id=doc_id, rank=rank, score=score)
            for rank, (doc_id, score) in enumerate(ranked[:k], start=1)
        ]


def save_dense_cache(
    documents: Sequence[Document],
    embeddings: np.ndarray,
    *,
    model_id: str,
    revision: str,
    max_length: int,
    directory: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Write the matrix and its labels. A mismatch is the loader's problem, not a silent rewrite."""
    if len(documents) != len({document.id for document in documents}):
        raise ValueError(f"重复的文档 id: {_duplicate_id([document.id for document in documents])}")
    matrix = _as_float_matrix(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != len(documents):
        raise ValueError("向量行数与文档数不一致")
    meta = EmbeddingCacheMeta(
        ids=[document.id for document in documents],
        text_sha256=[text_sha256(document.text) for document in documents],
        model_id=model_id,
        revision=revision,
        dim=int(matrix.shape[1]),
        max_length=max_length,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / EMBEDDINGS_NAME, matrix)
    (directory / META_NAME).write_text(
        json.dumps(meta.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_dense_cache(
    documents: Sequence[Document],
    *,
    model_id: str,
    revision: str,
    max_length: int,
    directory: Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """Load a cache only when ids, hashes, model, revision, max_length, and dim all match."""
    matrix_path = directory / EMBEDDINGS_NAME
    meta_path = directory / META_NAME
    if not matrix_path.is_file() or not meta_path.is_file():
        raise DenseCacheError(f"缓存不完整: {directory}")
    try:
        matrix = _as_float_matrix(np.load(matrix_path, allow_pickle=False))
        meta = EmbeddingCacheMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as exc:
        raise DenseCacheError(f"缓存无法读取: {directory}") from exc
    problems = _cache_problems(
        documents,
        matrix,
        meta,
        model_id=model_id,
        revision=revision,
        max_length=max_length,
    )
    if problems:
        raise DenseCacheError("；".join(problems))
    return matrix


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_problems(
    documents: Sequence[Document],
    matrix: np.ndarray,
    meta: EmbeddingCacheMeta,
    *,
    model_id: str,
    revision: str,
    max_length: int,
) -> list[str]:
    problems: list[str] = []
    ids = [document.id for document in documents]
    hashes = [text_sha256(document.text) for document in documents]
    if meta.ids != ids:
        problems.append("id 顺序不一致")
    if meta.text_sha256 != hashes:
        problems.append("文本 sha256 不一致")
    if meta.model_id != model_id:
        problems.append(f"模型标识不一致: 缓存 {meta.model_id}，当前 {model_id}")
    if meta.revision != revision:
        problems.append(f"模型 revision 不一致: 缓存 {meta.revision}，当前 {revision}")
    if meta.max_length != max_length:
        problems.append(f"max_length 不一致: 缓存 {meta.max_length}，当前 {max_length}")
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] != meta.dim:
        problems.append("向量形状与元数据不一致")
    return problems


def _as_float_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings)
    if matrix.dtype == object or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("向量必须是数值矩阵")
    return np.asarray(matrix, dtype=np.float64)


def _duplicate_id(ids: Sequence[str]) -> str:
    seen: set[str] = set()
    for doc_id in ids:
        if doc_id in seen:
            return doc_id
        seen.add(doc_id)
    return ""
