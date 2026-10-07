"""BM25 sparse retrieval. The caller supplies documents; this module does not read the corpus."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Sequence
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from retrieval.types import Document, Hit

# Module logger only. Segmentation uses a private Tokenizer, not jieba.lcut.
jieba.setLogLevel(logging.WARNING)

_STOPWORDS_PATH = Path(__file__).parent / "data" / "stopwords.txt"
_ENTITIES_PATH = Path(__file__).resolve().parents[1] / "corpus" / "manual_entities.json"
# Same frequency eval.fidelity writes into the global user dictionary for this name list.
_NAME_FREQ = 100_000
_PUNCTUATION = re.compile(r"^[\W_]+$", re.UNICODE)

_tokenizer: jieba.Tokenizer | None = None
_stopwords: frozenset[str] | None = None


def _manual_names() -> list[str]:
    payload = json.loads(_ENTITIES_PATH.read_text(encoding="utf-8"))
    return [str(item["name"]) for item in payload["entities"]]


def _segmenter() -> jieba.Tokenizer:
    """One private tokenizer shared by indexing and querying."""
    global _tokenizer
    if _tokenizer is None:
        tokenizer = jieba.Tokenizer()
        for name in _manual_names():
            tokenizer.add_word(name, freq=_NAME_FREQ)
        _tokenizer = tokenizer
    return _tokenizer


def _load_stopwords() -> frozenset[str]:
    global _stopwords
    if _stopwords is None:
        words: set[str] = set()
        for line in _STOPWORDS_PATH.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            words.add(item)
        if not words:
            raise ValueError(f"停用词表是空的: {_STOPWORDS_PATH}")
        _stopwords = frozenset(words)
    return _stopwords


def tokenize(text: str) -> list[str]:
    """Exact-mode tokens from the private tokenizer, without stopwords or punctuation."""
    stopwords = _load_stopwords()
    tokens: list[str] = []
    for raw in _segmenter().lcut(text):
        token = raw.strip()
        if not token or token in stopwords or _PUNCTUATION.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


class Bm25Index:
    """In-memory BM25Okapi index. Parameters k1, b, and epsilon stay at library defaults."""

    def __init__(self, documents: Sequence[Document]) -> None:
        seen: set[str] = set()
        stored: list[Document] = []
        for document in documents:
            if document.id in seen:
                raise ValueError(f"重复的文档 id: {document.id}")
            seen.add(document.id)
            stored.append(document)
        self._documents = stored
        self._tokens = [tokenize(document.text) for document in stored]
        self._model: BM25Okapi | None
        if any(self._tokens):
            self._model = BM25Okapi([list(tokens) for tokens in self._tokens])
        else:
            self._model = None

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(document.id for document in self._documents)

    def search(self, query: str, k: int, exclude_ids: Collection[str] = ()) -> list[Hit]:
        """Return up to `k` hits with score > 0. Ties break by id ascending."""
        if k <= 0:
            raise ValueError("k 必须是正整数")
        query_tokens = tokenize(query)
        if not query_tokens or self._model is None:
            return []
        raw_scores = [float(score) for score in self._model.get_scores(query_tokens)]
        excluded = set(exclude_ids)
        ranked = sorted(
            (
                (document.id, raw_scores[index])
                for index, document in enumerate(self._documents)
                if raw_scores[index] > 0.0 and document.id not in excluded
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            Hit(id=doc_id, rank=rank, score=score)
            for rank, (doc_id, score) in enumerate(ranked[:k], start=1)
        ]
