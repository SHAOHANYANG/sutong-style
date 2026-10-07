"""Generate the k = 0 control and the twelve few-shot groups from a retrieval plan.

The model is loaded once. Decoding goes through scripts.generate.greedy_decode so
the control stays comparable to corpus/generations/lora-eval59.jsonl. This module
must not import eval, stylometry, cn2an, or openai: the WSL2 environment does not
have those packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import structlog
from pydantic import BaseModel

from infra.qwen_prompt_tokenizer import QwenPromptTokenizer
from retrieval.prompt import (
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    Exemplar,
    assert_within_budget,
    build_prompt,
)
from scripts.generate import greedy_decode, load_inference_model, model_name
from scripts.train import Pair, load_pairs, select

LOGGER = structlog.get_logger()
FUSION_ORDER = ("equal", "balanced", "content", "style")
PIPELINE = "retrieval"


class ChatGenerator(Protocol):
    """One completion for a chat message list. Tests inject a fake."""

    def generate(self, messages: list[dict[str, str]]) -> str: ...


class TokenCounter(Protocol):
    """Prompt length in tokens. The real counter is QwenPromptTokenizer."""

    def count(self, messages: list[dict[str, str]]) -> int: ...


class PlanExemplar(BaseModel):
    id: str
    rank: int


class PlanQuery(BaseModel):
    id: str
    exemplars: list[PlanExemplar]


class Prepared(BaseModel):
    messages: list[dict[str, str]]
    digest: str
    exemplar_ids: list[str]
    fusion_config: str | None
    k: int


class UnslothGenerator:
    """The one real generator. torch and unsloth are imported by the loader."""

    def __init__(self, model: object, tokenizer: object, max_new_tokens: int) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        return greedy_decode(self._model, self._tokenizer, messages, self._max_new_tokens)


def jobs() -> list[tuple[str | None, int]]:
    """k = 0 first, then four configs times k = 1, 2, 3."""
    ordered: list[tuple[str | None, int]] = [(None, 0)]
    for config in FUSION_ORDER:
        for k in (1, 2, 3):
            ordered.append((config, k))
    return ordered


def group_key(config: str | None, k: int) -> str:
    if k == 0:
        return "retrieval-k0"
    if config is None:
        raise ValueError("k > 0 必须有融合配置")
    return f"retrieval-{config}-k{k}"


def prompt_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    """Hash the message list. The same list is the same greedy prompt."""
    payload = json.dumps(list(messages), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_case(
    query: Pair,
    config: str | None,
    k: int,
    plans: Mapping[str, Mapping[str, PlanQuery]],
    by_id: Mapping[str, Pair],
) -> Prepared:
    """Build the chat. Rank 1 is the last exemplar, matching build_prompt."""
    ranked: list[PlanExemplar] = []
    if k > 0:
        if config is None or config not in plans or query.id not in plans[config]:
            raise SystemExit(f"检索计划没有这条查询: {query.id} 配置 {config}")
        ranked = sorted(plans[config][query.id].exemplars, key=lambda item: (item.rank, item.id))
        ranked = ranked[:k]
        if len(ranked) != k:
            raise SystemExit(f"范例不足 {k} 条: 查询 {query.id} 配置 {config}")
    exemplars: list[Exemplar] = []
    for item in ranked:
        source = by_id.get(item.id)
        if source is None:
            raise SystemExit(f"范例不在 pairs 里: {item.id}")
        exemplars.append(
            Exemplar(id=item.id, vernacular=source.vernacular, original=source.original)
        )
    messages = build_prompt(query.vernacular, exemplars)
    return Prepared(
        messages=messages,
        digest=prompt_sha256(messages),
        exemplar_ids=[item.id for item in reversed(ranked)],
        fusion_config=config,
        k=k,
    )


def count_work(
    plans: Mapping[str, Mapping[str, PlanQuery]],
    queries: Sequence[Pair],
    by_id: Mapping[str, Pair],
) -> dict[str, object]:
    """Per-group row counts and how many distinct prompts a full run would decode."""
    seen: set[str] = set()
    rows_by_group: dict[str, int] = {}
    for config, k in jobs():
        key = group_key(config, k)
        rows_by_group[key] = len(queries)
        for query in queries:
            seen.add(prepare_case(query, config, k, plans, by_id).digest)
    return {
        "rows": len(queries) * len(jobs()),
        "rows_by_group": rows_by_group,
        "unique_prompts": len(seen),
    }


def run_sweep(
    *,
    plans: Mapping[str, Mapping[str, PlanQuery]],
    pairs: Sequence[Pair],
    baseline_path: Path,
    generator: ChatGenerator,
    counter: TokenCounter,
    output_dir: Path,
    manifest_path: Path,
    adapter: str,
    model: str,
    seed: int,
    plan_sha256: str,
    commit: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Write thirteen jsonl files. Stop before the twelve groups if k = 0 drifts."""
    queries = select(pairs, "eval")
    if not queries:
        raise SystemExit("eval 为空，无法扫描")
    by_id = {row.id: row for row in pairs}
    cache: dict[str, tuple[str, int]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, object] = {}
    for config, k in jobs():
        key = group_key(config, k)
        started = time.perf_counter()
        reused = _write_group(
            key=key,
            config=config,
            k=k,
            queries=queries,
            plans=plans,
            by_id=by_id,
            generator=generator,
            counter=counter,
            cache=cache,
            path=output_dir / f"{key}.jsonl",
            model=model,
            seed=seed,
            max_new_tokens=max_new_tokens,
            max_seq_length=max_seq_length,
        )
        if k == 0:
            _assert_k0(output_dir / f"{key}.jsonl", baseline_path, [row.id for row in queries])
        groups[key] = _group_record(
            path=output_dir / f"{key}.jsonl",
            config=config,
            k=k,
            reused=reused,
            seconds=time.perf_counter() - started,
        )
    manifest = {
        "adapter": adapter,
        "commit": commit,
        "groups": groups,
        "k0_check": {"baseline": str(baseline_path), "n": len(queries), "passed": True},
        "max_new_tokens": max_new_tokens,
        "max_seq_length": max_seq_length,
        "plan_sha256": plan_sha256,
        "seed": seed,
        "timestamp": timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("sweep_complete", groups=len(groups), manifest=str(manifest_path))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按检索计划做 k=0 与四种配置的 few-shot 生成")
    parser.add_argument("--plan", type=Path, default=Path("eval/reports/retrieval-plan.json"))
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("corpus/generations/lora-eval59.jsonl"),
    )
    parser.add_argument("--adapter", type=Path, default=Path("adapters/sutong-v2/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/reports/retrieval-sweep-manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不加载模型，只数每组条数和去重后的 prompt 数",
    )
    return parser.parse_args(None if argv is None else list(argv))


def main(
    argv: Sequence[str] | None = None,
    *,
    generator: ChatGenerator | None = None,
    counter: TokenCounter | None = None,
) -> int:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    pairs = load_pairs(args.pairs)
    plans = load_plan(args.plan)
    if args.dry_run:
        summary = count_work(plans, select(pairs, "eval"), {row.id: row for row in pairs})
        LOGGER.info(
            "sweep_dry_run",
            rows=summary["rows"],
            rows_by_group=summary["rows_by_group"],
            unique_prompts=summary["unique_prompts"],
        )
        return 0
    if generator is None or counter is None:
        if not args.adapter.exists():
            raise SystemExit(f"adapter 不存在: {args.adapter}")
        model, tokenizer = load_inference_model(args.adapter, args.max_seq_length, args.seed)
        generator = UnslothGenerator(model, tokenizer, args.max_new_tokens)
        counter = QwenPromptTokenizer(tokenizer)
    run_sweep(
        plans=plans,
        pairs=pairs,
        baseline_path=args.baseline,
        generator=generator,
        counter=counter,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        adapter=str(args.adapter),
        model=model_name(args.adapter),
        seed=args.seed,
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        commit=_commit(),
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
    )
    return 0


def load_plan(path: Path) -> dict[str, dict[str, PlanQuery]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("plans"), dict):
        raise SystemExit(f"检索计划缺少 plans: {path}")
    plans: dict[str, dict[str, PlanQuery]] = {}
    raw_plans = payload["plans"]
    for config in FUSION_ORDER:
        block = raw_plans.get(config)
        if not isinstance(block, list):
            raise SystemExit(f"检索计划缺少配置: {config}")
        plans[config] = {}
        for item in block:
            row = PlanQuery.model_validate(item)
            plans[config][row.id] = row
    return plans


def _write_group(
    *,
    key: str,
    config: str | None,
    k: int,
    queries: Sequence[Pair],
    plans: Mapping[str, Mapping[str, PlanQuery]],
    by_id: Mapping[str, Pair],
    generator: ChatGenerator,
    counter: TokenCounter,
    cache: dict[str, tuple[str, int]],
    path: Path,
    model: str,
    seed: int,
    max_new_tokens: int,
    max_seq_length: int,
) -> int:
    prepared = {query.id: prepare_case(query, config, k, plans, by_id) for query in queries}
    existing = _load_existing(path, prepared)
    for row in existing.values():
        _remember(cache, row)
    reused = 0
    for index, query in enumerate(queries, start=1):
        item = prepared[query.id]
        if query.id in existing:
            reused += 1
        elif item.digest in cache:
            output, tokens = cache[item.digest]
            _append(path, _row(query.id, model, seed, item, output, tokens))
            reused += 1
        else:
            try:
                tokens = counter.count(item.messages)
                assert_within_budget(
                    query.id,
                    tokens,
                    max_new_tokens=max_new_tokens,
                    max_seq_length=max_seq_length,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            output = generator.generate(item.messages)
            cache[item.digest] = (output, tokens)
            _append(path, _row(query.id, model, seed, item, output, tokens))
        if index % 10 == 0 or index == len(queries):
            LOGGER.info("sweep_progress", group=key, done=index, total=len(queries))
    got = {str(row["id"]) for row in _read_jsonl(path)}
    if got != {query.id for query in queries}:
        raise SystemExit(f"生成文件条数不对: {path.name}")
    return reused


def _load_existing(path: Path, prepared: Mapping[str, Prepared]) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    found: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        doc_id = row.get("id")
        if not isinstance(doc_id, str):
            raise SystemExit(f"生成文件行损坏: {path}")
        if doc_id in found:
            raise SystemExit(f"生成文件有重复 id: {doc_id}")
        if doc_id not in prepared:
            raise SystemExit(f"生成文件里有计划之外的 id: {doc_id}")
        if row.get("prompt_sha256") != prepared[doc_id].digest:
            raise SystemExit(f"prompt_sha256 不一致: {doc_id} {path.name}")
        found[doc_id] = row
    return found


def _remember(cache: dict[str, tuple[str, int]], row: Mapping[str, object]) -> None:
    digest = row.get("prompt_sha256")
    output = row.get("output")
    tokens = row.get("prompt_tokens")
    if not isinstance(digest, str) or not isinstance(output, str) or not isinstance(tokens, int):
        raise SystemExit("已有生成行缺少 prompt_sha256、output 或 prompt_tokens")
    previous = cache.get(digest)
    if previous is not None and previous != (output, tokens):
        raise SystemExit(f"同一 prompt 有两份不同输出: {digest}")
    cache[digest] = (output, tokens)


def _row(
    doc_id: str,
    model: str,
    seed: int,
    item: Prepared,
    output: str,
    tokens: int,
) -> dict[str, object]:
    return {
        "exemplar_ids": list(item.exemplar_ids),
        "fusion_config": item.fusion_config,
        "id": doc_id,
        "k": item.k,
        "model": model,
        "output": output,
        "pipeline": PIPELINE,
        "prompt_sha256": item.digest,
        "prompt_tokens": tokens,
        "seed": seed,
        "trace": None,
    }


def _append(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SystemExit(f"生成文件行不是对象: {path}")
        rows.append({str(key): value for key, value in payload.items()})
    return rows


def _assert_k0(path: Path, baseline_path: Path, expected_ids: Sequence[str]) -> None:
    if not baseline_path.is_file():
        raise SystemExit(f"基线生成不存在: {baseline_path}")
    baseline: dict[str, str] = {}
    for row in _read_jsonl(baseline_path):
        doc_id = row.get("id")
        output = row.get("output")
        if not isinstance(doc_id, str) or not isinstance(output, str):
            raise SystemExit(f"基线行损坏: {baseline_path}")
        baseline[doc_id] = output
    got = {str(row["id"]): str(row["output"]) for row in _read_jsonl(path)}
    if set(got) != set(expected_ids) or set(got) != set(baseline):
        raise SystemExit("k=0 与基线的样本 id 不一致")
    for doc_id, output in got.items():
        if output != baseline[doc_id]:
            raise SystemExit(f"k=0 输出与基线不同: {doc_id}")


def _group_record(
    *,
    path: Path,
    config: str | None,
    k: int,
    reused: int,
    seconds: float,
) -> dict[str, object]:
    rows = _read_jsonl(path)
    tokens: list[int] = []
    for row in rows:
        value = row.get("prompt_tokens")
        if isinstance(value, int):
            tokens.append(value)
    return {
        "fusion_config": config,
        "k": k,
        "n": len(rows),
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_tokens": _spread(tokens),
        "reused": reused,
        "seconds": seconds,
    }


def _spread(values: Sequence[int]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise SystemExit("没有 prompt token 数，无法写分布")
    return {
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "std": float(array.std(ddof=0)),
    }


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
