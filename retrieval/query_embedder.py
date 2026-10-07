"""Look up dense query vectors by text hash. A miss is an error, never a zero vector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import ValidationError

from retrieval.dense import (
    EMBEDDINGS_NAME,
    META_NAME,
    DenseCacheError,
    EmbeddingCacheMeta,
    text_sha256,
)


class CachedQueryEmbedder:
    """Vectors encoded by the same model, revision, and max_length as the document index."""

    def __init__(self, query_dir: Path, index_dir: Path) -> None:
        query_meta, matrix = _load_query(query_dir)
        index_meta = _load_meta(index_dir)
        _require_same_encoder(query_meta, index_meta)
        self._dim = int(matrix.shape[1])
        self._rows = {
            digest: np.asarray(matrix[index], dtype=np.float64)
            for index, digest in enumerate(query_meta.text_sha256)
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float64)
        rows: list[np.ndarray] = []
        for text in texts:
            digest = text_sha256(text)
            row = self._rows.get(digest)
            if row is None:
                raise ValueError(f"查询缓存没有这条文本的向量: {digest}")
            rows.append(row)
        return np.vstack(rows)


def _require_same_encoder(query_meta: EmbeddingCacheMeta, index_meta: EmbeddingCacheMeta) -> None:
    if query_meta.model_id != index_meta.model_id:
        raise DenseCacheError(
            f"模型标识不一致: 查询缓存 {query_meta.model_id}，索引缓存 {index_meta.model_id}"
        )
    if query_meta.revision != index_meta.revision:
        raise DenseCacheError(
            f"模型 revision 不一致: 查询缓存 {query_meta.revision}，索引缓存 {index_meta.revision}"
        )
    if query_meta.max_length != index_meta.max_length:
        raise DenseCacheError(
            f"max_length 不一致: 查询缓存 {query_meta.max_length}，索引缓存 {index_meta.max_length}"
        )


def _load_query(directory: Path) -> tuple[EmbeddingCacheMeta, np.ndarray]:
    meta = _load_meta(directory)
    matrix_path = directory / EMBEDDINGS_NAME
    if not matrix_path.is_file():
        raise DenseCacheError(f"缓存不完整: {directory}")
    try:
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise DenseCacheError(f"缓存无法读取: {directory}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(meta.text_sha256) or matrix.shape[1] != meta.dim:
        raise DenseCacheError("向量形状与元数据不一致")
    return meta, matrix


def _load_meta(directory: Path) -> EmbeddingCacheMeta:
    path = directory / META_NAME
    if not path.is_file():
        raise DenseCacheError(f"缓存不完整: {directory}")
    try:
        return EmbeddingCacheMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise DenseCacheError(f"缓存无法读取: {directory}") from exc
