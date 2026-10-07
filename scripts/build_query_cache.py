"""Encode eval vernaculars for dense queries. Run the real mode on the GPU machine."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import structlog

from retrieval.dense import save_dense_cache
from retrieval.types import Document
from scripts.split_corpus import CorpusSplit
from scripts.train import Pair, load_pairs, select

LOGGER = structlog.get_logger()
DEFAULT_OUTPUT = Path("retrieval/data/query_cache")


def eval_query_documents(pairs: Sequence[Pair], split: CorpusSplit) -> list[Document]:
    """Eval vernaculars in pairs.jsonl order, after the same split cross-check as the index."""
    seen: set[str] = set()
    train_ids = set(split.train)
    eval_ids = set(split.eval)
    for pair in pairs:
        if pair.id in seen:
            raise ValueError(f"重复的文档 id: {pair.id}")
        seen.add(pair.id)
        in_train = pair.id in train_ids
        in_eval = pair.id in eval_ids
        if pair.split == "train" and in_eval:
            raise ValueError(f"id 出现在 split.eval: {pair.id}")
        if pair.split == "train" and not in_train:
            raise ValueError(f"id 不在 split.train: {pair.id}")
        if in_train and not in_eval:
            expected = "train"
        elif in_eval and not in_train:
            expected = "eval"
        else:
            expected = None
        if pair.split != expected:
            raise ValueError(f"split 字段与 split.json 不一致: {pair.id}")
    return [Document(id=pair.id, text=pair.vernacular) for pair in select(pairs, "eval")]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 bge-m3 为 eval 白话建查询向量缓存")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None, help="默认 BAAI/bge-m3")
    parser.add_argument("--revision", default=None, help="不传则用 Hub 返回的 commit hash")
    parser.add_argument("--max-length", type=int, default=None, help="默认 512")
    parser.add_argument("--batch-size", type=int, default=None, help="默认 8")
    parser.add_argument("--device", default=None, help="默认有 CUDA 用 cuda，否则 cpu")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对 pairs 与 split，打印条数后退出，不下载模型、不写文件",
    )
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    documents = eval_query_documents(
        load_pairs(args.pairs),
        CorpusSplit.model_validate_json(args.split.read_text(encoding="utf-8")),
    )
    if args.dry_run:
        from infra.bge_embedder import DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH, DEFAULT_MODEL_ID

        LOGGER.info(
            "query_cache_dry_run",
            documents=len(documents),
            output=str(args.output_dir),
            model_id=args.model or DEFAULT_MODEL_ID,
            max_length=DEFAULT_MAX_LENGTH if args.max_length is None else args.max_length,
            batch_size=DEFAULT_BATCH_SIZE if args.batch_size is None else args.batch_size,
        )
        return
    from infra.bge_embedder import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MAX_LENGTH,
        DEFAULT_MODEL_ID,
        BgeEmbedder,
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
    event = "query_cache_truncated" if embedder.truncated else "query_cache_written"
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
