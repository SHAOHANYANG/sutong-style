"""Run the four pre-registered agent arms on the eval split (SPEC 4.8).

The model is loaded once. Round one is never decoded again: its prompt is the
same list the k sweep already decoded, so the output is looked up by
prompt_sha256 in the two sweep files. A first-round prompt that is missing there
stops the run before the model loads.

The verifier and the scorer run inside the loop, so this module imports eval and
stylometry and has to load on WSL2. It must not import openai, sacrebleu, or
eval.judge: that environment does not have them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog
from pydantic import BaseModel

from agent.config import AgentConfig
from agent.graph import run_agent
from agent.nodes import Scorer, Verifier
from agent.prompts import KIND_ORDER, FeedbackFormat
from agent.scorer import PredictorScorer, fit_train_predictor
from agent.state import MAX_GENERATIONS, AgentResult
from agent.verifier import FidelityVerifier
from eval.loaders import load_config, load_gazetteer, load_style_reference
from infra.qwen_prompt_tokenizer import QwenPromptTokenizer
from retrieval.prompt import (
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    Exemplar,
    assert_within_budget,
    build_prompt,
)
from scripts.generate import load_inference_model, model_name
from scripts.sweep_topk import (
    ChatGenerator,
    PlanQuery,
    Prepared,
    TokenCounter,
    UnslothGenerator,
    group_key,
    load_plan,
    prepare_case,
    prompt_sha256,
)
from scripts.train import Pair, load_pairs, select

LOGGER = structlog.get_logger()
PIPELINE = "agent"
# Exemplar source name -> (fusion config, k) of the sweep group round one comes from.
SOURCES: dict[str, tuple[str | None, int]] = {"balanced-k2": ("balanced", 2), "k0": (None, 0)}
FEEDBACK_FORMATS: tuple[FeedbackFormat, ...] = ("followup", "restate")
PRIMARY_ARM = "agent-balanced-k2-followup"
TERMINATIONS = ("accepted", "max_rounds", "recursion_limit", "fallback")


class Arm(BaseModel):
    """One exemplar source crossed with one feedback format."""

    source: str
    feedback_format: FeedbackFormat

    @property
    def key(self) -> str:
        return f"agent-{self.source}-{self.feedback_format}"

    @property
    def fusion_config(self) -> str | None:
        return SOURCES[self.source][0]

    @property
    def k(self) -> int:
        return SOURCES[self.source][1]

    @property
    def baseline_key(self) -> str:
        return group_key(self.fusion_config, self.k)


class PlanRetriever:
    """Exemplars fixed by the retrieval plan. The input text is looked up, never searched."""

    def __init__(self, table: Mapping[str, Sequence[Exemplar]]) -> None:
        self._table = {text: list(items) for text, items in table.items()}

    def retrieve(self, vernacular: str) -> list[Exemplar]:
        if vernacular not in self._table:
            raise SystemExit("检索计划里没有这段输入")
        return list(self._table[vernacular])


class CachedGenerator:
    """Look a prompt up by sha before decoding. An over-budget prompt stops the run."""

    def __init__(
        self,
        inner: ChatGenerator | None,
        counter: TokenCounter | None,
        cache: dict[str, str],
        *,
        max_new_tokens: int,
        max_seq_length: int,
    ) -> None:
        self._inner = inner
        self._counter = counter
        self._cache = cache
        self._max_new_tokens = max_new_tokens
        self._max_seq_length = max_seq_length
        self.sample_id = ""
        self.calls = 0
        self.hits = 0
        self.tokens: list[int] = []

    def begin(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.calls = 0
        self.hits = 0
        self.tokens = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        digest = prompt_sha256(messages)
        if digest in self._cache:
            self.hits += 1
            return self._cache[digest]
        if self._inner is None or self._counter is None:
            raise SystemExit(f"样本 {self.sample_id}：需要真实生成，但没有加载模型")
        # SystemExit, not ValueError: the generate node turns ordinary exceptions into a
        # fallback result, and an over-budget prompt must stop the run instead.
        try:
            tokens = self._counter.count(messages)
            assert_within_budget(
                self.sample_id,
                tokens,
                max_new_tokens=self._max_new_tokens,
                max_seq_length=self._max_seq_length,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        output = self._inner.generate(messages)
        self._cache[digest] = output
        self.calls += 1
        self.tokens.append(tokens)
        return output


def arms() -> list[Arm]:
    """The primary arm first, then the three exploratory ones."""
    return [
        Arm(source=source, feedback_format=feedback_format)
        for source in SOURCES
        for feedback_format in FEEDBACK_FORMATS
    ]


def round_distribution(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    """How many cases ended after 1, 2, 3 rounds, which round was kept, and why it stopped."""
    total: Counter[str] = Counter({str(index): 0 for index in range(1, MAX_GENERATIONS + 1)})
    selected: Counter[str] = Counter({str(index): 0 for index in range(1, MAX_GENERATIONS + 1)})
    selected["none"] = 0
    termination: Counter[str] = Counter({name: 0 for name in TERMINATIONS})
    for row in rows:
        total[str(row["total_rounds"])] += 1
        picked = row["selected_round"]
        selected["none" if picked is None else str(picked)] += 1
        termination[str(row["termination"])] += 1
    return {
        "selected_round": dict(sorted(selected.items())),
        "termination": dict(sorted(termination.items())),
        "total_rounds": dict(sorted(total.items())),
    }


def violation_transitions(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    """Per kind: violations in round one, in the kept round, fixed, and newly introduced.

    A violation is identified by (kind, expected, actual). A fallback that kept no
    round counts as having no final violations; the fallback itself is listed elsewhere.
    """
    table: dict[str, dict[str, int]] = {
        kind: {
            "final": 0,
            "final_cases": 0,
            "first": 0,
            "first_cases": 0,
            "fixed": 0,
            "introduced": 0,
        }
        for kind in KIND_ORDER
    }
    for row in rows:
        rounds = _rounds(row)
        first = _violation_keys(rounds[0]) if rounds else set()
        picked = row["selected_round"]
        final = _violation_keys(rounds[picked - 1]) if isinstance(picked, int) else set()
        for kind in KIND_ORDER:
            before = {item for item in first if item[0] == kind}
            after = {item for item in final if item[0] == kind}
            cell = table[kind]
            cell["first"] += len(before)
            cell["final"] += len(after)
            cell["fixed"] += len(before - after)
            cell["introduced"] += len(after - before)
            cell["first_cases"] += int(bool(before))
            cell["final_cases"] += int(bool(after))
    return table


def first_round_violation_cases(rows: Sequence[Mapping[str, object]]) -> int:
    """Cases the loop had a reason to revise. The rest equal the control by construction."""
    return sum(1 for row in rows if _rounds(row) and _violation_keys(_rounds(row)[0]))


def run_agent_eval(
    *,
    plans: Mapping[str, Mapping[str, PlanQuery]],
    pairs: Sequence[Pair],
    baselines_dir: Path,
    generator: ChatGenerator,
    counter: TokenCounter,
    verifier: Verifier,
    scorer: Scorer,
    output_dir: Path,
    manifest_path: Path,
    adapter: str,
    model: str,
    seed: int,
    plan_sha256: str,
    commit: str,
    scorer_info: Mapping[str, object] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Write four jsonl files and the manifest. Finished cases are skipped on restart."""
    queries = select(pairs, "eval")
    if not queries:
        raise SystemExit("eval 为空，无法运行 agent")
    by_id = {row.id: row for row in pairs}
    baselines = load_baselines(baselines_dir)
    prepared = {arm.key: check_first_round(arm, queries, plans, by_id, baselines) for arm in arms()}
    cache = first_round_cache(baselines)
    cached = CachedGenerator(
        generator,
        counter,
        cache,
        max_new_tokens=max_new_tokens,
        max_seq_length=max_seq_length,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, object] = {}
    for arm in arms():
        path = output_dir / f"{arm.key}.jsonl"
        _run_arm(
            arm=arm,
            queries=queries,
            prepared=prepared[arm.key],
            by_id=by_id,
            baseline=baselines[arm.baseline_key],
            generator=cached,
            verifier=verifier,
            scorer=scorer,
            path=path,
            model=model,
            seed=seed,
        )
        records[arm.key] = _arm_record(arm, path, baselines[arm.baseline_key])
    manifest: dict[str, object] = {
        "adapter": adapter,
        "arms": records,
        "baselines": {
            key: {
                "path": str(baselines_dir / f"{key}.jsonl"),
                "sha256": hashlib.sha256((baselines_dir / f"{key}.jsonl").read_bytes()).hexdigest(),
            }
            for key in sorted(baselines)
        },
        "commit": commit,
        "decoding": "greedy",
        "max_generations": AgentConfig().max_generations,
        "max_new_tokens": max_new_tokens,
        "max_seq_length": max_seq_length,
        "model": model,
        "plan_sha256": plan_sha256,
        "primary_arm": PRIMARY_ARM,
        "re_retrieve_score_threshold": AgentConfig().re_retrieve_score_threshold,
        "scorer": dict(scorer_info or {}),
        "seed": seed,
        "timestamp": timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("agent_eval_complete", arms=len(records), manifest=str(manifest_path))
    return manifest


def dry_run(
    *,
    plans: Mapping[str, Mapping[str, PlanQuery]],
    pairs: Sequence[Pair],
    baselines_dir: Path,
    verifier: Verifier,
) -> dict[str, object]:
    """Check the plan, the pairs and both sweep files; count the cases that need the model."""
    queries = select(pairs, "eval")
    if not queries:
        raise SystemExit("eval 为空，无法运行 agent")
    by_id = {row.id: row for row in pairs}
    baselines = load_baselines(baselines_dir)
    rows_by_arm: dict[str, int] = {}
    needing: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for arm in arms():
        check_first_round(arm, queries, plans, by_id, baselines)
        rows_by_arm[arm.key] = len(queries)
        kinds: Counter[str] = Counter({kind: 0 for kind in KIND_ORDER})
        count = 0
        for query in queries:
            output = str(baselines[arm.baseline_key][query.id]["output"])
            violations = verifier.verify(query.vernacular, output) if output.strip() else []
            count += int(bool(violations) or not output.strip())
            kinds.update(item.kind for item in violations)
        needing[arm.key] = count
        by_kind[arm.key] = dict(kinds)
    summary: dict[str, object] = {
        "first_round_violation_cases": needing,
        "first_round_violations_by_kind": by_kind,
        # Each revised case decodes at most max_generations - 1 more times.
        "max_model_calls": {
            key: value * (AgentConfig().max_generations - 1) for key, value in needing.items()
        },
        "rows_by_arm": rows_by_arm,
    }
    LOGGER.info("agent_dry_run", **summary)
    return summary


def load_baselines(directory: Path) -> dict[str, dict[str, dict[str, object]]]:
    """The two sweep files round one is taken from, keyed by group and then by case id."""
    loaded: dict[str, dict[str, dict[str, object]]] = {}
    for key in sorted({arm.baseline_key for arm in arms()}):
        path = directory / f"{key}.jsonl"
        if not path.is_file():
            raise SystemExit(f"基线生成不存在: {path}")
        rows: dict[str, dict[str, object]] = {}
        for row in _read_jsonl(path):
            doc_id = row.get("id")
            if (
                not isinstance(doc_id, str)
                or not isinstance(row.get("output"), str)
                or not isinstance(row.get("prompt_sha256"), str)
                or not isinstance(row.get("prompt_tokens"), int)
            ):
                raise SystemExit(f"基线行损坏: {path}")
            if doc_id in rows:
                raise SystemExit(f"基线有重复 id: {doc_id} {path.name}")
            rows[doc_id] = row
        loaded[key] = rows
    return loaded


def first_round_cache(
    baselines: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, str]:
    """prompt_sha256 -> output for every row of the sweep files."""
    cache: dict[str, str] = {}
    for rows in baselines.values():
        for row in rows.values():
            digest = str(row["prompt_sha256"])
            output = str(row["output"])
            if cache.get(digest, output) != output:
                raise SystemExit(f"同一 prompt 有两份不同输出: {digest}")
            cache[digest] = output
    return cache


def check_first_round(
    arm: Arm,
    queries: Sequence[Pair],
    plans: Mapping[str, Mapping[str, PlanQuery]],
    by_id: Mapping[str, Pair],
    baselines: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, Prepared]:
    """SPEC 4.8: every first-round prompt must already be in the sweep file, sha for sha."""
    prepared: dict[str, Prepared] = {}
    for query in queries:
        item = prepare_case(query, arm.fusion_config, arm.k, plans, by_id)
        row = baselines[arm.baseline_key].get(query.id)
        if row is None or row["prompt_sha256"] != item.digest:
            raise SystemExit(
                f"第一轮 prompt 在基线里找不到对应的 sha: {query.id} {arm.baseline_key}"
            )
        if build_prompt(query.vernacular, _exemplars(item, by_id)) != item.messages:
            raise SystemExit(f"agent 第一轮消息与扫描不一致: {query.id} {arm.key}")
        prepared[query.id] = item
    return prepared


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按预注册跑 agent 的四个臂（SPEC 4.8）")
    parser.add_argument("--plan", type=Path, default=Path("eval/reports/retrieval-plan.json"))
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument("--baselines-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument("--adapter", type=Path, default=Path("adapters/sutong-v2/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/reports/agent-run-manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不加载模型，核对计划与基线，数每臂第一轮就有违规的样本",
    )
    return parser.parse_args(None if argv is None else list(argv))


def main(
    argv: Sequence[str] | None = None,
    *,
    generator: ChatGenerator | None = None,
    counter: TokenCounter | None = None,
    scorer: Scorer | None = None,
) -> int:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    pairs = load_pairs(args.pairs)
    plans = load_plan(args.plan)
    config = load_config(args.config)
    verifier = FidelityVerifier(load_gazetteer(Path(config.gazetteer)))
    if args.dry_run:
        dry_run(plans=plans, pairs=pairs, baselines_dir=args.baselines_dir, verifier=verifier)
        return 0
    scorer_info: dict[str, object] = {"kind": "injected"}
    if scorer is None:
        reference = load_style_reference(config)
        train = select(pairs, "train")
        predictor = fit_train_predictor(
            reference,
            [row.vernacular for row in train],
            [row.original for row in train],
            [row.work for row in train],
        )
        scorer = PredictorScorer(reference, predictor)
        scorer_info = {"alpha": predictor.alpha, "kind": "predictor_b", "n_train": len(train)}
    # Fail on a missing first-round prompt before the model is loaded.
    dry_run(plans=plans, pairs=pairs, baselines_dir=args.baselines_dir, verifier=verifier)
    if generator is None or counter is None:
        if not args.adapter.exists():
            raise SystemExit(f"adapter 不存在: {args.adapter}")
        model, tokenizer = load_inference_model(args.adapter, args.max_seq_length, args.seed)
        generator = UnslothGenerator(model, tokenizer, args.max_new_tokens)
        counter = QwenPromptTokenizer(tokenizer)
    run_agent_eval(
        plans=plans,
        pairs=pairs,
        baselines_dir=args.baselines_dir,
        generator=generator,
        counter=counter,
        verifier=verifier,
        scorer=scorer,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        adapter=str(args.adapter),
        model=model_name(args.adapter),
        seed=args.seed,
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        commit=_commit(),
        scorer_info=scorer_info,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
    )
    return 0


def _run_arm(
    *,
    arm: Arm,
    queries: Sequence[Pair],
    prepared: Mapping[str, Prepared],
    by_id: Mapping[str, Pair],
    baseline: Mapping[str, Mapping[str, object]],
    generator: CachedGenerator,
    verifier: Verifier,
    scorer: Scorer,
    path: Path,
    model: str,
    seed: int,
) -> None:
    existing = _load_existing(path, prepared, arm)
    exemplars = {query.id: _exemplars(prepared[query.id], by_id) for query in queries}
    retriever = PlanRetriever({query.vernacular: exemplars[query.id] for query in queries})
    config = AgentConfig(feedback_format=arm.feedback_format)
    for index, query in enumerate(queries, start=1):
        if query.id not in existing:
            generator.begin(query.id)
            started = time.perf_counter()
            result = run_agent(
                query.vernacular,
                retriever=retriever,
                generator=generator,
                verifier=verifier,
                scorer=scorer,
                config=config,
            )
            seconds = time.perf_counter() - started
            if not result.rounds or result.rounds[0].output != baseline[query.id]["output"]:
                raise SystemExit(f"第一轮输出与基线不同: {query.id} {arm.key}")
            if result.termination == "fallback":
                LOGGER.warning(
                    "agent_fallback",
                    arm=arm.key,
                    id=query.id,
                    fallback_kind=result.fallback_kind,
                )
            _append(
                path,
                _row(query.id, arm, prepared[query.id], result, generator, model, seed, seconds),
            )
        if index % 10 == 0 or index == len(queries):
            LOGGER.info("agent_progress", arm=arm.key, done=index, total=len(queries))
    got = {str(row["id"]) for row in _read_jsonl(path)}
    if got != {query.id for query in queries}:
        raise SystemExit(f"生成文件条数不对: {path.name}")


def _row(
    doc_id: str,
    arm: Arm,
    item: Prepared,
    result: AgentResult,
    generator: CachedGenerator,
    model: str,
    seed: int,
    seconds: float,
) -> dict[str, object]:
    return {
        "cache_hits": generator.hits,
        "error": result.error,
        "exemplar_ids": list(item.exemplar_ids),
        "exemplar_source": arm.source,
        "fallback_kind": result.fallback_kind,
        "feedback_format": arm.feedback_format,
        "first_prompt_sha256": item.digest,
        "fusion_config": item.fusion_config,
        "id": doc_id,
        "k": item.k,
        "model": model,
        "model_calls": generator.calls,
        "output": result.output,
        "pipeline": PIPELINE,
        "revision_prompt_tokens": list(generator.tokens),
        "rounds": [record.model_dump() for record in result.rounds],
        "seconds": seconds,
        "seed": seed,
        "selected_round": result.selected_round,
        "termination": result.termination,
        "total_rounds": result.total_rounds,
        "trace": [event.model_dump() for event in result.trace],
    }


def _arm_record(
    arm: Arm,
    path: Path,
    baseline: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows = _read_jsonl(path)
    revision_tokens = [value for row in rows for value in _int_list(row, "revision_prompt_tokens")]
    first_tokens = [_int(baseline[str(row["id"])], "prompt_tokens") for row in rows]
    return {
        "baseline": arm.baseline_key,
        "cache_hits": sum(_int(row, "cache_hits") for row in rows),
        "exemplar_source": arm.source,
        "fallbacks": [
            {"fallback_kind": row.get("fallback_kind"), "id": row["id"]}
            for row in rows
            if row["termination"] == "fallback"
        ],
        "feedback_format": arm.feedback_format,
        "first_round_violation_cases": first_round_violation_cases(rows),
        "model_calls": sum(_int(row, "model_calls") for row in rows),
        "n": len(rows),
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_tokens": {
            "first_round": _spread(first_tokens),
            "revision": _spread(revision_tokens),
        },
        "rounds": round_distribution(rows),
        "seconds": sum(_seconds(row) for row in rows),
    }


def _load_existing(
    path: Path,
    prepared: Mapping[str, Prepared],
    arm: Arm,
) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        doc_id = row.get("id")
        if not isinstance(doc_id, str):
            raise SystemExit(f"生成文件行损坏: {path}")
        if doc_id in found:
            raise SystemExit(f"生成文件有重复 id: {doc_id}")
        if doc_id not in prepared:
            raise SystemExit(f"生成文件里有计划之外的 id: {doc_id}")
        if row.get("first_prompt_sha256") != prepared[doc_id].digest:
            raise SystemExit(f"prompt_sha256 不一致: {doc_id} {path.name}")
        if row.get("feedback_format") != arm.feedback_format:
            raise SystemExit(f"反馈格式不一致: {doc_id} {path.name}")
        found[doc_id] = row
    return found


def _exemplars(item: Prepared, by_id: Mapping[str, Pair]) -> list[Exemplar]:
    """Rank order. Prepared.exemplar_ids is prompt order, which is the reverse."""
    return [
        Exemplar(id=doc_id, vernacular=by_id[doc_id].vernacular, original=by_id[doc_id].original)
        for doc_id in reversed(item.exemplar_ids)
    ]


def _rounds(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    rounds = row.get("rounds")
    if not isinstance(rounds, list) or not all(isinstance(item, dict) for item in rounds):
        raise SystemExit(f"rounds 损坏: {row.get('id')}")
    return list(rounds)


def _violation_keys(record: Mapping[str, object]) -> set[tuple[str, str | None, str | None]]:
    violations = record.get("violations")
    if not isinstance(violations, list):
        raise SystemExit("rounds 里的 violations 损坏")
    keys: set[tuple[str, str | None, str | None]] = set()
    for item in violations:
        if not isinstance(item, dict) or item.get("kind") not in KIND_ORDER:
            raise SystemExit("违规记录损坏")
        keys.add((str(item["kind"]), item.get("expected"), item.get("actual")))
    return keys


def _int(row: Mapping[str, object], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{name} 不是整数: {row.get('id')}")
    return value


def _seconds(row: Mapping[str, object]) -> float:
    value = row.get("seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"seconds 不是数: {row.get('id')}")
    return float(value)


def _int_list(row: Mapping[str, object], name: str) -> list[int]:
    value = row.get(name)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise SystemExit(f"{name} 损坏: {row.get('id')}")
    return list(value)


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


def _spread(values: Sequence[int]) -> dict[str, float] | None:
    """None when nothing was decoded, so an empty arm is visible instead of an error."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "n": float(array.size),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "std": float(array.std(ddof=0)),
    }


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
