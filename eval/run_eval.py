"""Score a generation file. The judge is optional and stays off the network when skipped."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
import yaml
from pydantic import BaseModel, Field

from eval.fidelity import assess
from eval.judge import (
    Completer,
    DiskCache,
    Outcome,
    case_score,
    compare_case,
    style_win_rate,
    vote_tally,
)
from scripts.vernacularize import text_similarity
from stylometry.distance import StyleReference
from stylometry.lexicon import LiteraryLexicon

LOGGER = structlog.get_logger()
AGGREGATE_FIELDS = (
    "style_distance",
    "entity_recall",
    "numeral_recall",
    "hallucination_rate",
    "style_win_rate",
    "copy_ratio",
    "ppl",
)


class EvalConfig(BaseModel):
    """Committed baseline settings. Paths are relative to the working directory."""

    pipeline: str
    model: str
    seed: int
    cases: str
    style_reference: str
    lexicon: str
    gazetteer: str
    judge_model: str
    judge_prompt: str
    opponent: Literal["vernacular"]


class EvalCase(BaseModel):
    id: str
    vernacular: str
    original: str


class Generation(BaseModel):
    id: str
    output: str
    model: str = "unknown"
    pipeline: str = "baseline"
    seed: int = 42


class CaseReport(BaseModel):
    id: str
    metrics: dict[str, float | None]
    output_hash: str


class EvalReport(BaseModel):
    run_id: str
    timestamp: str
    pipeline: str
    model: str
    config: dict[str, object]
    n_cases: int
    aggregate: dict[str, float | None]
    distribution: dict[str, dict[str, float | int] | None]
    per_case: list[CaseReport] = Field(default_factory=list)


def load_config(path: Path) -> EvalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是映射：{path}")
    return EvalConfig.model_validate(payload)


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase.model_validate_json(line))
    return cases


def load_generations(path: Path) -> dict[str, Generation]:
    generations: dict[str, Generation] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = Generation.model_validate_json(line)
        generations[row.id] = row
    return generations


def load_style_reference(config: EvalConfig) -> StyleReference:
    lexicon = LiteraryLexicon.model_validate_json(Path(config.lexicon).read_text(encoding="utf-8"))
    payload = json.loads(Path(config.style_reference).read_text(encoding="utf-8"))
    return StyleReference(
        mean=np.array(payload["mean"], dtype=np.float64),
        std=np.array(payload["std"], dtype=np.float64),
        lexicon=lexicon,
    )


def load_gazetteer(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload["words"]
    if not isinstance(words, list):
        raise ValueError("gazetteer words 不是列表")
    return {word for word in words if isinstance(word, str)}


def output_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    """Mean plus the spread. The aggregate field itself keeps the mean."""
    array = np.asarray(values, dtype=np.float64)
    spread: dict[str, float | int] = {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    return spread


def judge_model_name(config: EvalConfig) -> str:
    """SPEC 3.3: JUDGE_MODEL overrides the name in the config."""
    return os.environ.get("JUDGE_MODEL", config.judge_model)


def build_judge_completer(model: str) -> Completer:
    """Real client. --skip-judge must not call this."""
    from dotenv import load_dotenv

    from infra.openai_generator import OpenAICompatibleGenerator

    load_dotenv()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is missing")
    return OpenAICompatibleGenerator(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL"),
        model=model,
    )


def evaluate(
    config: EvalConfig,
    generations_path: Path,
    run_id: str,
    *,
    skip_judge: bool,
    report_path: Path,
    cache_dir: Path,
    completer: Completer | None = None,
) -> EvalReport:
    """Score every case. Missing generations are an error, not a silent drop."""
    cases = load_cases(Path(config.cases))
    generations = load_generations(generations_path)
    missing = [case.id for case in cases if case.id not in generations]
    if missing:
        raise ValueError("生成文件缺少样本：" + ", ".join(missing))
    reference = load_style_reference(config)
    gazetteer = load_gazetteer(Path(config.gazetteer))
    template = Path(config.judge_prompt).read_text(encoding="utf-8")
    model = judge_model_name(config)
    cache = DiskCache(cache_dir)
    judge = completer
    if not skip_judge and judge is None:
        judge = build_judge_completer(model)

    per_case: list[CaseReport] = []
    distances: list[float] = []
    entities: list[float] = []
    numerals: list[float] = []
    hallucinations: list[float] = []
    copy_ratios: list[float] = []
    outcomes: list[Outcome] = []
    for case in cases:
        generation = generations[case.id]
        report = assess(case.vernacular, generation.output, gazetteer)
        distance = reference.distance(generation.output)
        copied = text_similarity(generation.output, case.vernacular)
        outcome: Outcome | None = None
        if judge is not None:
            outcome = compare_case(
                judge,
                cache,
                template,
                model,
                case.original,
                generation.output,
                case.vernacular,
            )
            outcomes.append(outcome)
        metrics: dict[str, float | None] = {
            "style_distance": distance,
            "entity_recall": report.entity_recall,
            "numeral_recall": report.numeral_recall,
            "hallucination_rate": report.hallucination_rate,
            "style_win_rate": case_score(outcome) if outcome else None,
            "copy_ratio": copied,
            "ppl": None,
        }
        distances.append(distance)
        copy_ratios.append(copied)
        entities.append(report.entity_recall)
        numerals.append(report.numeral_recall)
        hallucinations.append(report.hallucination_rate)
        per_case.append(
            CaseReport(id=case.id, metrics=metrics, output_hash=output_hash(generation.output))
        )

    win_rate = style_win_rate(outcomes) if outcomes else None
    copy_summary = summarize(copy_ratios)
    measured: dict[str, float | None] = {
        "style_distance": summarize(distances)["mean"],
        "entity_recall": summarize(entities)["mean"],
        "numeral_recall": summarize(numerals)["mean"],
        "hallucination_rate": summarize(hallucinations)["mean"],
        "style_win_rate": win_rate,
        "copy_ratio": copy_summary["mean"],
        "ppl": None,
    }
    aggregate = {name: measured[name] for name in AGGREGATE_FIELDS}
    distribution: dict[str, dict[str, float | int] | None] = {
        "style_distance": summarize(distances),
        "entity_recall": summarize(entities),
        "numeral_recall": summarize(numerals),
        "hallucination_rate": summarize(hallucinations),
        "style_win_rate": vote_tally(outcomes) if outcomes else None,
        "copy_ratio": copy_summary,
        "ppl": None,
    }
    models = {generations[case.id].model for case in cases}
    report_model = next(iter(models)) if len(models) == 1 else config.model
    body = EvalReport(
        run_id=run_id,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pipeline=config.pipeline,
        model=report_model,
        config={
            **config.model_dump(),
            "judge_model_used": model,
            "skip_judge": skip_judge,
            "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        },
        n_cases=len(cases),
        aggregate=aggregate,
        distribution=distribution,
        per_case=per_case,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(body.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "eval_report_written",
        run_id=run_id,
        n_cases=len(cases),
        skip_judge=skip_judge,
        report=str(report_path),
    )
    return body


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="对生成结果计算保真、风格距离和可选的风格胜率")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("eval/.judge_cache"))
    args = parser.parse_args(argv)
    report_path = args.report or Path("eval/reports") / f"{args.run_id}.json"
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    evaluate(
        load_config(args.config),
        args.generations,
        args.run_id,
        skip_judge=bool(args.skip_judge),
        report_path=report_path,
        cache_dir=args.cache_dir,
        completer=None,
    )


if __name__ == "__main__":
    main()
