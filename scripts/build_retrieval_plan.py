"""Write the fused top-3 exemplars for every eval query and fusion config.

Retrieval is deterministic and does not need a GPU. The later generation step
reads this plan instead of searching again. The plan contains ids and numbers
only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog
import yaml
from pydantic import BaseModel

from eval.metrics import profile_gap_per_case
from eval.run_eval import load_config, load_style_reference, summarize
from retrieval.bm25 import Bm25Index
from retrieval.dense import (
    EMBEDDINGS_NAME,
    META_NAME,
    DenseIndex,
    EmbeddingCacheMeta,
    load_dense_cache,
)
from retrieval.hybrid import DEFAULT_DEPTH, DEFAULT_RRF_K, ROUTE_NAMES, HybridRetriever
from retrieval.query_embedder import CachedQueryEmbedder
from retrieval.style_index import StyleIndex
from retrieval.style_predictor import fit_predictor
from retrieval.types import FusedHit
from scripts.build_dense_index import train_documents
from scripts.split_corpus import CorpusSplit
from scripts.train import Pair, load_pairs, select
from stylometry.distance import StyleReference

LOGGER = structlog.get_logger()
TOP_K = 3


class RouteWeights(BaseModel):
    bm25: float
    dense: float
    style: float


class RetrievalPlanConfig(BaseModel):
    embedding_model: str
    embedding_revision: str
    max_length: int
    depth: int
    rrf_k: int
    primary_config: str
    primary_k: int
    configs: dict[str, RouteWeights]


def load_plan_config(path: Path) -> RetrievalPlanConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是映射：{path}")
    return RetrievalPlanConfig.model_validate(payload)


def jaccard(left: set[str], right: set[str]) -> float:
    """Size of the intersection divided by the size of the union."""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def validate_hits(
    config_name: str,
    query_id: str,
    hits: Sequence[FusedHit],
    eval_ids: set[str],
) -> None:
    """Stop the run when a selection breaks the plan rules."""
    if len(hits) != TOP_K:
        raise SystemExit(
            f"范例不足 {TOP_K} 条: 查询 {query_id} 配置 {config_name} 实际 {len(hits)} 条"
        )
    for hit in hits:
        if hit.id in eval_ids:
            raise SystemExit(
                f"范例 id 在 split.eval 中: {hit.id} 查询 {query_id} 配置 {config_name}"
            )
        if hit.id == query_id:
            raise SystemExit(f"范例是查询自己: {query_id} 配置 {config_name}")
        sources = set(hit.sources)
        if config_name == "content" and "style" in sources:
            raise SystemExit(f"content 配置的范例含 style: 查询 {query_id} 范例 {hit.id}")
        if config_name == "style" and sources != {"style"}:
            raise SystemExit(f"style 配置的范例不是只来自 style: 查询 {query_id} 范例 {hit.id}")


def build_report(
    *,
    plan_config: RetrievalPlanConfig,
    reference: StyleReference,
    pairs: Sequence[Pair],
    split: CorpusSplit,
    index_dir: Path,
    query_dir: Path,
    commit: str,
    timestamp: str,
) -> dict[str, object]:
    _require_query_cache(query_dir)
    documents = train_documents(pairs, split)
    eval_rows = select(pairs, "eval")
    train_rows = select(pairs, "train")
    if not documents or not eval_rows:
        raise SystemExit("训练集或 eval 为空，无法生成检索计划")
    if [row.id for row in train_rows] != [document.id for document in documents]:
        raise SystemExit("train 原文顺序与 pairs.jsonl 不一致")
    index_meta = _load_meta(index_dir)
    query_meta = _load_meta(query_dir)
    matrix = load_dense_cache(
        documents,
        model_id=plan_config.embedding_model,
        revision=plan_config.embedding_revision,
        max_length=plan_config.max_length,
        directory=index_dir,
    )
    embedder = CachedQueryEmbedder(query_dir, index_dir)
    bm25 = Bm25Index(documents)
    dense = DenseIndex([document.id for document in documents], matrix, embedder)
    original_z = _stack(reference, [row.original for row in train_rows])
    predictor = fit_predictor(
        _stack(reference, [row.vernacular for row in train_rows]),
        original_z,
        [row.work for row in train_rows],
    )
    style = StyleIndex([row.id for row in train_rows], original_z)
    train_z = {row.id: original_z[index] for index, row in enumerate(train_rows)}
    train_work = {row.id: row.work for row in train_rows}
    query_z = {row.id: reference.zscore(row.original) for row in eval_rows}
    eval_ids = set(split.eval)
    weights = {
        name: {"bm25": item.bm25, "dense": item.dense, "style": item.style}
        for name, item in plan_config.configs.items()
    }
    plans: dict[str, list[dict[str, object]]] = {}
    held: dict[str, list[tuple[Pair, list[FusedHit]]]] = {}
    for name in sorted(weights):
        retriever = HybridRetriever(
            bm25,
            dense,
            style,
            reference,
            predictor,
            depth=plan_config.depth,
            rrf_k=plan_config.rrf_k,
            weights=weights[name],
        )
        records: list[dict[str, object]] = []
        chosen: list[tuple[Pair, list[FusedHit]]] = []
        for query in eval_rows:
            hits = retriever.search(query.vernacular, TOP_K, exclude_ids=[query.id])
            validate_hits(name, query.id, hits, eval_ids)
            records.append(
                {
                    "exemplars": [_hit_record(hit) for hit in hits],
                    "id": query.id,
                    "work": query.work,
                }
            )
            chosen.append((query, hits))
        plans[name] = records
        held[name] = chosen
    return {
        "caches": {
            "index": _cache_record(index_meta),
            "query": _cache_record(query_meta),
        },
        "commit": commit,
        "config": plan_config.model_dump(),
        "jaccard": {f"k{k}": _jaccard_table(plans, k) for k in (1, 2, 3)},
        "parameters": {
            "depth": plan_config.depth,
            "k": TOP_K,
            "n_eval": len(eval_rows),
            "n_train": len(train_rows),
            "primary_config": plan_config.primary_config,
            "primary_k": plan_config.primary_k,
            "rrf_k": plan_config.rrf_k,
        },
        "plans": plans,
        "summary": {
            name: _summary(chosen, train_z, train_work, query_z) for name, chosen in held.items()
        },
        "timestamp": timestamp,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 eval 查询写四套融合配置的检索计划")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/retrieval.yaml"))
    parser.add_argument("--style-config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument("--index-dir", type=Path, default=Path("retrieval/data"))
    parser.add_argument("--query-dir", type=Path, default=Path("retrieval/data/query_cache"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/retrieval-plan.json"))
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    _require_query_cache(args.query_dir)
    plan_config = load_plan_config(args.config)
    if plan_config.depth != DEFAULT_DEPTH or plan_config.rrf_k != DEFAULT_RRF_K:
        raise SystemExit(f"配置里的 depth/rrf_k 不是预注册值 {DEFAULT_DEPTH}/{DEFAULT_RRF_K}")
    reference = load_style_reference(load_config(args.style_config))
    report = build_report(
        plan_config=plan_config,
        reference=reference,
        pairs=load_pairs(args.pairs),
        split=CorpusSplit.model_validate_json(args.split.read_text(encoding="utf-8")),
        index_dir=args.index_dir,
        query_dir=args.query_dir,
        commit=_commit(),
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parameters = report["parameters"]
    n_eval = parameters["n_eval"] if isinstance(parameters, dict) else 0
    n_train = parameters["n_train"] if isinstance(parameters, dict) else 0
    LOGGER.info("retrieval_plan_written", n_eval=n_eval, n_train=n_train, output=str(args.output))


def _require_query_cache(directory: Path) -> None:
    if not (directory / META_NAME).is_file() or not (directory / EMBEDDINGS_NAME).is_file():
        raise SystemExit(f"查询缓存不存在: {directory}")


def _load_meta(directory: Path) -> EmbeddingCacheMeta:
    payload = (directory / META_NAME).read_text(encoding="utf-8")
    return EmbeddingCacheMeta.model_validate_json(payload)


def _cache_record(meta: EmbeddingCacheMeta) -> dict[str, object]:
    return {
        "created_at": meta.created_at,
        "max_length": meta.max_length,
        "model_id": meta.model_id,
        "revision": meta.revision,
    }


def _stack(reference: StyleReference, texts: Sequence[str]) -> np.ndarray:
    return np.vstack([reference.zscore(text) for text in texts])


def _hit_record(hit: FusedHit) -> dict[str, object]:
    return {
        "id": hit.id,
        "rank": hit.rank,
        "score": hit.score,
        "sources": {name: int(hit.sources[name]) for name in sorted(hit.sources)},
    }


def _summary(
    chosen: Sequence[tuple[Pair, list[FusedHit]]],
    train_z: Mapping[str, np.ndarray],
    train_work: Mapping[str, str],
    query_z: Mapping[str, np.ndarray],
) -> dict[str, object]:
    return {f"k{k}": _k_summary(chosen, train_z, train_work, query_z, k) for k in (1, 2, 3)}


def _k_summary(
    chosen: Sequence[tuple[Pair, list[FusedHit]]],
    train_z: Mapping[str, np.ndarray],
    train_work: Mapping[str, str],
    query_z: Mapping[str, np.ndarray],
    k: int,
) -> dict[str, object]:
    works = sorted({query.work for query, _hits in chosen})
    return {
        "by_work": {
            work: _block(
                [(query, hits) for query, hits in chosen if query.work == work],
                train_z,
                train_work,
                query_z,
                k,
            )
            for work in works
        },
        **_block(chosen, train_z, train_work, query_z, k),
    }


def _block(
    chosen: Sequence[tuple[Pair, list[FusedHit]]],
    train_z: Mapping[str, np.ndarray],
    train_work: Mapping[str, str],
    query_z: Mapping[str, np.ndarray],
    k: int,
) -> dict[str, object]:
    slots = 0
    route_hits = {name: 0 for name in ROUTE_NAMES}
    style_only = 0
    content_only = 0
    same_work = 0
    gaps: list[float] = []
    for query, hits in chosen:
        taken = list(hits)[:k]
        slot_gaps: list[float] = []
        for hit in taken:
            slots += 1
            sources = set(hit.sources)
            for name in ROUTE_NAMES:
                if name in sources:
                    route_hits[name] += 1
            if sources == {"style"}:
                style_only += 1
            if "style" not in sources and ("bm25" in sources or "dense" in sources):
                content_only += 1
            if train_work.get(hit.id) == query.work:
                same_work += 1
            delta = train_z[hit.id] - query_z[query.id]
            slot_gaps.append(float(profile_gap_per_case(delta.reshape(1, -1))[0]))
        gaps.append(float(np.mean(slot_gaps)))
    return {
        "content_only": _rate(content_only, slots),
        "profile_gap": summarize(gaps),
        "route_rate": {name: _rate(route_hits[name], slots) for name in ROUTE_NAMES},
        "same_work": _rate(same_work, slots),
        "style_only": _rate(style_only, slots),
    }


def _rate(count: int, total: int) -> float:
    if total == 0:
        raise SystemExit("没有范例，无法计算比例")
    return count / total


def _jaccard_table(plans: Mapping[str, Sequence[Mapping[str, object]]], k: int) -> dict[str, float]:
    names = sorted(plans)
    table: dict[str, float] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            scores = [
                jaccard(_id_set(left_row, k), _id_set(right_row, k))
                for left_row, right_row in zip(plans[left_name], plans[right_name], strict=True)
            ]
            table[f"{left_name}|{right_name}"] = float(np.mean(scores))
    return table


def _id_set(row: Mapping[str, object], k: int) -> set[str]:
    exemplars = row["exemplars"]
    if not isinstance(exemplars, list):
        raise SystemExit(f"范例列表损坏: {row.get('id')}")
    return {str(item["id"]) for item in exemplars[:k] if isinstance(item, Mapping)}


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
