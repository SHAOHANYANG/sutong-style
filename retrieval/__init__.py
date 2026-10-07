"""Retrieval indexes.

StyleIndex and HybridRetriever are loaded on demand. Importing this package must
not import stylometry, because the query-cache script is run in an environment
that does not have that stack's extra packages, and stylometry is not needed to
encode a query vector.
"""

from typing import TYPE_CHECKING, Any

from retrieval.bm25 import Bm25Index, tokenize
from retrieval.dense import DenseCacheError, DenseIndex, load_dense_cache, save_dense_cache
from retrieval.fusion import fuse
from retrieval.query_embedder import CachedQueryEmbedder
from retrieval.types import Document, Embedder, FusedHit, Hit

if TYPE_CHECKING:
    from retrieval.hybrid import HybridRetriever
    from retrieval.style_index import StyleIndex

__all__ = [
    "Bm25Index",
    "CachedQueryEmbedder",
    "DenseCacheError",
    "DenseIndex",
    "Document",
    "Embedder",
    "FusedHit",
    "Hit",
    "HybridRetriever",
    "StyleIndex",
    "fuse",
    "load_dense_cache",
    "save_dense_cache",
    "tokenize",
]


def __getattr__(name: str) -> Any:
    if name == "StyleIndex":
        from retrieval.style_index import StyleIndex

        return StyleIndex
    if name == "HybridRetriever":
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
