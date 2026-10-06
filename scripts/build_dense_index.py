"""Build the dense cache on the GPU machine. Tests must not import the embedder."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import structlog

from retrieval.dense import save_dense_cache
from retrieval.types import Document
from scripts.chunk_corpus import CorpusChunk
from scripts.split_corpus import CorpusSplit, load_chunks

LOGGER = structlog.get_logger()


def train_documents(chunks: Sequence[CorpusChunk], split: CorpusSplit) -> list[Document]:
    """Originals for every train id, in split.json order. Eval ids are left out."""
    if len(split.train) != len(set(split.train)):
        raise ValueError("split.train 含重复 id")
    by_id: dict[str, CorpusChunk] = {}
    for chunk in chunks:
        if chunk.id in by_id:
            raise ValueError(f"重复的 chunk id: {chunk.id}")
        by_id[chunk.id] = chunk
    known = set(split.train) | set(split.eval)
    unknown = sorted(set(by_id).difference(known))
    if unknown:
        raise ValueError(f"chunk 不在 split.json 里: {unknown}")
    missing = [chunk_id for chunk_id in split.train if chunk_id not in by_id]
    if missing:
        raise ValueError(f"train id 在 chunks 里缺失: {missing}")
    return [Document(id=chunk_id, text=by_id[chunk_id].original) for chunk_id in split.train]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 bge-m3 为 train 原文建 dense 缓存")
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("retrieval/data"))
    parser.add_argument("--model", default=None, help="默认 BAAI/bge-m3")
    parser.add_argument("--revision", default=None, help="不传则用 Hub 返回的 commit hash")
    parser.add_argument("--max-length", type=int, default=None, help="默认 512")
    parser.add_argument("--batch-size", type=int, default=None, help="默认 8")
    parser.add_argument("--device", default=None, help="默认有 CUDA 用 cuda，否则 cpu")
    return parser.parse_args(None if argv is None else list(argv))


def main() -> None:
    from infra.bge_embedder import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MAX_LENGTH,
        DEFAULT_MODEL_ID,
        BgeEmbedder,
    )

    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args()
    documents = train_documents(
        load_chunks(args.chunks),
        CorpusSplit.model_validate_json(args.split.read_text(encoding="utf-8")),
    )
    embedder = BgeEmbedder(
        args.model or DEFAULT_MODEL_ID,
        revision=args.revision,
        max_length=DEFAULT_MAX_LENGTH if args.max_length is None else args.max_length,
        batch_size=DEFAULT_BATCH_SIZE if args.batch_size is None else args.batch_size,
        device=args.device,
    )
    vectors = embedder.encode([document.text for document in documents])
    save_dense_cache(
        documents,
        vectors,
        model_id=embedder.model_id,
        revision=embedder.revision,
        max_length=embedder.max_length,
        directory=args.output_dir,
    )
    event = "dense_index_truncated" if embedder.truncated else "dense_index_written"
    LOGGER.info(
        event,
        documents=len(documents),
        dim=int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        model_id=embedder.model_id,
        revision=embedder.revision,
        max_length=embedder.max_length,
        truncated=embedder.truncated,
        output=str(args.output_dir),
    )


if __name__ == "__main__":
    main()
