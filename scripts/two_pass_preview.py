"""Preregistered 20-case, two-call design experiment; no full-run entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from scripts.chunk_corpus import CorpusChunk
from scripts.rebuild_metrics import TextMetrics, entity_candidates, final_failures, measure
from scripts.rebuild_reporting import (
    AttemptArtifact,
    Distribution,
    ModelIdentityMismatchError,
    ProviderMetadata,
    RunMetadata,
    SamplingParameters,
    describe,
)
from scripts.vernacularize import (
    PROMPT_DIR,
    Generator,
    RebuildReport,
    VernacularExample,
    load_chunks,
    load_style_examples,
    stratified_sample,
)

LOGGER = structlog.get_logger()
APPROVED_REPLACEMENTS_PATH = Path("scripts/data/approved_replacement_terms.json")
TWO_PASS_DIR = PROMPT_DIR / "two_pass"
METRICS = (
    "pinc1",
    "pinc2",
    "pinc3",
    "pinc4",
    "pinc6",
    "pinc_mean_1_to_4",
    "sbleu",
    "gram6",
    "gram4",
    "similarity",
    "length_ratio",
    "entity_recall",
    "numeral_recall",
)


class RepairIssue(BaseModel):
    category: Literal["entity", "numeral"]
    lost_or_changed_fact: str
    repair: str


class RepairReply(BaseModel):
    issues: list[RepairIssue]
    repaired_text: str = Field(min_length=1)


class ApprovedReplacementTerms(BaseModel):
    """Manually reviewed examples for residue checks; category rules remain prompt-only."""

    version: str
    source_document: str
    terms: list[str]


class StageObservation(BaseModel):
    id: str
    work: str
    stage: int
    seed: int
    called_at_utc: str
    prompt_sha256: str
    output_sha256: str | None = None
    raw_response_sha256: str | None = None
    provider: ProviderMetadata = Field(default_factory=ProviderMetadata)
    metrics: TextMetrics | None = None
    errors: list[str] = Field(default_factory=list)


class StageArtifact(BaseModel):
    observation: StageObservation
    original: str
    prompt: str
    raw_output: str | None = None
    text: str | None = None


class CaseResult(BaseModel):
    id: str
    work: str
    selected_example_ids: list[str]
    first: StageObservation | None = None
    second: StageObservation | None = None
    repair_issues: list[RepairIssue] = Field(default_factory=list)
    delta6: float | None = None
    delta4: float | None = None
    delta_pinc6: float | None = None
    delta_pinc4: float | None = None
    accepted_by_deterministic_gates: bool = False
    reasons: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    expected_count: int
    missing_ids: list[str]
    overall: dict[str, Distribution]
    by_work: dict[str, dict[str, Distribution]]


class ReviewStatus(BaseModel):
    status: str = "pending_human_review"
    conclusion: str | None = None


class TwoPassReport(BaseModel):
    status: str
    metadata: RunMetadata
    preflight_reports: list[str] = Field(
        default_factory=lambda: ["corpus/rebuild_report_two_pass_preflight_20261003.json"]
    )
    preregistration: str = (
        "SPEC 1.5.2; manual lexicon review and category rules approved before preview"
    )
    metric_definitions: dict[str, str] = Field(
        default_factory=lambda: {
            "PINC": "Chinese-only candidate occurrences; binary source membership; macro mean",
            "PINC-1..4": "Arithmetic mean of orders 1,2,3,4; per-order PINC also reported",
            "sBLEU": (
                "Chinese characters; sentence BLEU 0-100; sacrebleu 2.6.0; "
                "tokenize=none, exp smoothing, effective_order=True; macro mean"
            ),
            "SequenceMatcher": "Historical difflib ratio, original/output including punctuation",
        }
    )
    reference: MetricSummary
    comparisons: dict[str, MetricSummary]
    cases: list[CaseResult]
    artifacts_path: str
    review_sample_ids: list[str]
    human_review: ReviewStatus = Field(default_factory=ReviewStatus)
    assistant_review: list[str] = Field(default_factory=list)
    interpretation: list[str] = Field(
        default_factory=lambda: [
            "Historical N-gram metrics are retrospective, not the historical preregistered gate.",
            "Second-pass self-reported repairs are not independent semantic-fidelity evidence.",
            "No full corpus generation is authorized; "
            "deterministic gates do not substitute for human review.",
        ]
    )


def metric_summary(rows: list[tuple[str, str, TextMetrics | None]]) -> MetricSummary:
    overall: dict[str, Distribution] = {}
    grouped: dict[str, dict[str, Distribution]] = {}
    for key in METRICS:
        values = [
            float(getattr(row, key))
            for _, _, row in rows
            if row is not None and getattr(row, key) is not None
        ]
        overall[key] = describe(values)
        for work in sorted({work for _, work, _ in rows}):
            group = [
                float(getattr(row, key))
                for _, w, row in rows
                if w == work and row is not None and getattr(row, key) is not None
            ]
            grouped.setdefault(work, {})[key] = describe(group)
    return MetricSummary(
        expected_count=len(rows),
        missing_ids=[
            case_id
            for case_id, _, row in rows
            if row is None or row.pinc6 is None or row.pinc4 is None
        ],
        overall=overall,
        by_work=grouped,
    )


def render_first(
    original: str, examples: list[VernacularExample], terms: list[str], entities: list[str]
) -> str:
    example_template = (PROMPT_DIR / "example.txt").read_text(encoding="utf-8")
    demonstration = (
        (PROMPT_DIR / "examples.txt")
        .read_text(encoding="utf-8")
        .format(
            examples="\n".join(
                example_template.format(index=i, original=row.original, vernacular=row.vernacular)
                for i, row in enumerate(examples, start=1)
            )
        )
    )
    instruction = (TWO_PASS_DIR / "destroy.txt").read_text(encoding="utf-8")
    return demonstration + instruction.format(
        original=original,
        banned_terms="、".join(terms) or "无命中",
        entities="、".join(entities) or "无候选",
    )


def render_second(original: str, draft: str, terms: list[str], entities: list[str]) -> str:
    return (
        (TWO_PASS_DIR / "repair.txt")
        .read_text(encoding="utf-8")
        .format(
            original=original,
            draft=draft,
            banned_terms="、".join(terms) or "无命中",
            entities="、".join(entities) or "无候选",
        )
    )


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    return (
        "\n".join(raw.splitlines()[1:-1]).strip()
        if raw.startswith("```") and raw.endswith("```")
        else raw
    )


def run_experiment(
    chunks: list[CorpusChunk],
    baseline: RebuildReport,
    examples: list[VernacularExample],
    approved_terms: list[str],
    generator: Generator,
    metadata: RunMetadata,
    report_path: Path,
    artifacts_path: Path,
) -> tuple[list[CaseResult], str]:
    """Two fixed calls per case, retain every output, never retry rejected generations."""
    if len(chunks) != 20 or metadata.retries != 0:
        raise ValueError("Only one 20-case no-retry experiment is permitted")
    old_records = {row.id: row for row in baseline.per_case}
    by_example = {row.id: row for row in examples}
    banned = approved_terms
    lock, aborted = threading.Lock(), threading.Event()
    results: list[CaseResult] = []
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)

    def archive(artifact: StageArtifact) -> None:
        with lock, artifacts_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(artifact.model_dump_json() + "\n")

    def invoke(chunk: CorpusChunk, prompt: str, stage: int, seed: int) -> StageArtifact:
        observation = StageObservation(
            id=chunk.id,
            work=chunk.work,
            stage=stage,
            seed=seed,
            called_at_utc=datetime.now(UTC).isoformat(),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
        artifact = StageArtifact(observation=observation, original=chunk.original, prompt=prompt)
        try:
            if aborted.is_set():
                observation.errors.append("cancelled_after_model_mismatch")
                return artifact
            artifact.raw_output = generator.generate(
                prompt,
                seed=seed,
                temperature=metadata.sampling.temperature,
                top_p=metadata.sampling.top_p,
                max_tokens=metadata.sampling.max_tokens,
                thinking_mode=metadata.sampling.thinking_mode,
            )
            observation.raw_response_sha256 = hashlib.sha256(
                artifact.raw_output.encode("utf-8")
            ).hexdigest()
            if observation_metadata := getattr(generator, "response_metadata", None):
                provider_value: object = observation_metadata()
                if isinstance(provider_value, ProviderMetadata):
                    observation.provider = provider_value
            if observation.provider.finish_reason == "length":
                observation.errors.append("provider_output_truncated")
        except ModelIdentityMismatchError:
            aborted.set()
            observation.errors.append("model_identity_mismatch")
            if observation_metadata := getattr(generator, "response_metadata", None):
                provider_value = observation_metadata()
                if isinstance(provider_value, ProviderMetadata):
                    observation.provider = provider_value
        except Exception as exc:
            observation.errors.append(type(exc).__name__)
        return artifact

    def process(chunk: CorpusChunk) -> CaseResult:
        old = old_records[chunk.id]
        case = CaseResult(
            id=chunk.id, work=chunk.work, selected_example_ids=old.selected_example_ids
        )
        if aborted.is_set():
            case.reasons.append("cancelled_after_model_mismatch")
            return case
        entities = entity_candidates(chunk.original)
        terms = [word for word in banned if word in chunk.original and word not in entities]
        first_prompt = render_first(
            chunk.original, [by_example[i] for i in old.selected_example_ids], terms, entities
        )
        first = invoke(chunk, first_prompt, 1, old.seed)
        case.first = first.observation
        if first.raw_output is not None:
            first.text = strip_fences(first.raw_output)
            first.observation.output_sha256 = hashlib.sha256(first.text.encode("utf-8")).hexdigest()
            first.observation.metrics = measure(chunk.original, first.text, terms, entities)
        archive(first)
        if first.text is None or aborted.is_set():
            case.reasons = [*first.observation.errors, "second_skipped_without_valid_first_call"]
            return case
        second = invoke(
            chunk, render_second(chunk.original, first.text, terms, entities), 2, old.seed
        )
        case.second = second.observation
        if second.raw_output is not None:
            try:
                reply = RepairReply.model_validate_json(strip_fences(second.raw_output))
                second.text = reply.repaired_text.strip()
                case.repair_issues = reply.issues
                second.observation.output_sha256 = hashlib.sha256(
                    second.text.encode("utf-8")
                ).hexdigest()
                second.observation.metrics = measure(chunk.original, second.text, terms, entities)
            except ValidationError:
                second.observation.errors.append("repair_json_invalid")
        archive(second)
        case.reasons = first.observation.errors + second.observation.errors
        if first.observation.metrics and second.observation.metrics:
            before, after = first.observation.metrics, second.observation.metrics
            case.delta6 = (
                after.gram6 - before.gram6
                if after.gram6 is not None and before.gram6 is not None
                else None
            )
            case.delta4 = (
                after.gram4 - before.gram4
                if after.gram4 is not None and before.gram4 is not None
                else None
            )
            case.delta_pinc6 = (
                after.pinc6 - before.pinc6
                if after.pinc6 is not None and before.pinc6 is not None
                else None
            )
            case.delta_pinc4 = (
                after.pinc4 - before.pinc4
                if after.pinc4 is not None and before.pinc4 is not None
                else None
            )
            case.reasons.extend(final_failures(before, after))
        else:
            case.reasons.append("missing_stage_metrics")
        case.accepted_by_deterministic_gates = not case.reasons
        return case

    with ThreadPoolExecutor(max_workers=metadata.concurrency) as executor:
        futures = [executor.submit(process, chunk) for chunk in chunks]
        for future in as_completed(futures):
            case = future.result()
            results.append(case)
            LOGGER.info(
                "two_pass_case",
                id=case.id,
                first=case.first.metrics.model_dump()
                if case.first and case.first.metrics
                else None,
                second=case.second.metrics.model_dump()
                if case.second and case.second.metrics
                else None,
                reasons=case.reasons,
            )
            # Keep completed cases visible even if the process is subsequently interrupted.
            report_path.write_text(
                json.dumps(
                    {
                        "status": "in_progress",
                        "metadata": metadata.model_dump(),
                        "cases": [row.model_dump() for row in results],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return sorted(
        results, key=lambda row: row.id
    ), "aborted_model_mismatch" if aborted.is_set() else "complete"


def main() -> int:
    parser = argparse.ArgumentParser(description="一次20条两段式预览，严禁全量")
    parser.add_argument(
        "--report", type=Path, default=Path("corpus/rebuild_report_two_pass_20261003.json")
    )
    args = parser.parse_args()
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    artifacts_path = Path("corpus/generations") / (args.report.stem + "_stages.jsonl")
    if args.report.exists() or artifacts_path.exists():
        raise ValueError("No overwrite, resume, or second experiment is permitted at these paths")
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "scripts",
            "infra",
            "docs/SPEC.md",
            "pyproject.toml",
            "uv.lock",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError(
            "Commit preregistration, prompts, lexicon and implementation before external calls"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    flash_path = Path("corpus/rebuild_report_round2_20261003.json")
    pro_path = Path("corpus/rebuild_report_model_control_20261003.json")
    flash = RebuildReport.model_validate_json(flash_path.read_text(encoding="utf-8"))
    pro = RebuildReport.model_validate_json(pro_path.read_text(encoding="utf-8"))
    if flash.metadata is None or pro.metadata is None or pro.artifacts_path is None:
        raise ValueError("Baseline provenance is missing")
    metadata = pro.metadata.model_copy(deep=True)
    if (
        metadata.model != "deepseek-v4-pro"
        or metadata.sampling != SamplingParameters(thinking_mode="disabled")
        or metadata.retries != 0
        or metadata.concurrency != 4
    ):
        raise ValueError("Frozen sampling parameters or model changed")
    if metadata.selected_ids != flash.metadata.selected_ids or len(metadata.selected_ids) != 20:
        raise ValueError("Fixed 20-case baseline changed")
    for source, digest in metadata.source_sha256.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {source}")
    approved = ApprovedReplacementTerms.model_validate_json(
        APPROVED_REPLACEMENTS_PATH.read_text(encoding="utf-8")
    )
    if len(approved.terms) != 67 or len(set(approved.terms)) != 67:
        raise ValueError("Approved replacement catalog must contain exactly 67 unique terms")
    all_chunks = {row.id: row for row in load_chunks(Path("corpus/chunks.jsonl"))}
    chunks = [all_chunks[case_id] for case_id in metadata.selected_ids]
    examples = load_style_examples(Path("corpus/salvaged_pairs.jsonl"))
    pro_artifacts = [
        AttemptArtifact.model_validate_json(line)
        for line in Path(pro.artifacts_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pro_by_id = {row.id: row for row in pro.per_case}
    flash_by_id = {row.id: row for row in flash.per_case}
    for artifact in pro_artifacts:
        row = artifact.record
        old = flash_by_id[row.id]
        if (
            row.provider.model != "deepseek-v4-pro"
            or artifact.original != all_chunks[row.id].original
            or (row.seed, row.selected_example_ids) != (old.seed, old.selected_example_ids)
        ):
            raise ValueError("Source, model or seed/few-shot baseline identity changed")
    if set(pro_by_id) != set(metadata.selected_ids) or len(pro_artifacts) != 20:
        raise ValueError("Baseline observations incomplete")
    review_ids = [row.id for row in stratified_sample(chunks, count=6, seed=42)]
    metadata.started_at_utc = datetime.now(UTC).isoformat()
    metadata.completed_at_utc = None
    metadata.prompt_git_commit = commit
    metadata.code_git_commit = commit
    metadata.prompt_path = "scripts/prompts/two_pass/{destroy,repair}.txt"
    prompt_paths = [
        PROMPT_DIR / "example.txt",
        PROMPT_DIR / "examples.txt",
        TWO_PASS_DIR / "destroy.txt",
        TWO_PASS_DIR / "repair.txt",
        APPROVED_REPLACEMENTS_PATH,
    ]
    metadata.prompt_sha256 = hashlib.sha256(
        b"".join(path.read_bytes() for path in prompt_paths)
    ).hexdigest()
    metadata.source_sha256.update(
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [flash_path, pro_path, APPROVED_REPLACEMENTS_PATH]
        }
    )
    metadata.mode = "two_pass_design_preview"
    metadata.control_baseline = str(pro_path)
    metadata.control_baseline_sha256 = hashlib.sha256(pro_path.read_bytes()).hexdigest()
    metadata.control_verification = (
        "Same 20 cases, originals, per-case seeds, stage1 examples and sampling; "
        "bundled preregistered design changes, NOT a single-variable experiment."
    )
    load_dotenv()
    endpoint = os.environ.get("LLM_BASE_URL", "")
    if urlsplit(endpoint).hostname != "api.deepseek.com":
        raise ValueError("Only api.deepseek.com is authorized")
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is missing")
    from infra.openai_generator import OpenAICompatibleGenerator

    generator = OpenAICompatibleGenerator(api_key=api_key, base_url=endpoint, model=metadata.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "metadata": metadata.model_dump(),
                "review_sample_ids": review_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("two_pass_preflight", cases=20, max_calls=40, review_sample_ids=review_ids)
    cases, status = run_experiment(
        chunks, pro, examples, approved.terms, generator, metadata, args.report, artifacts_path
    )
    metadata.completed_at_utc = datetime.now(UTC).isoformat()
    banned = approved.terms
    comparisons: dict[str, MetricSummary] = {}
    for label, report in (("flash_round2", flash), ("pro_control", pro)):
        if report.artifacts_path is None:
            raise ValueError("Missing historical artifacts")
        history = [
            AttemptArtifact.model_validate_json(line)
            for line in Path(report.artifacts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        comparisons[label] = metric_summary(
            [
                (
                    row.record.id,
                    row.record.work,
                    measure(row.original, row.output, banned, entity_candidates(row.original))
                    if row.output is not None
                    else None,
                )
                for row in history
            ]
        )
    comparisons["first_pass"] = metric_summary(
        [(row.id, row.work, row.first.metrics if row.first else None) for row in cases]
    )
    comparisons["second_pass"] = metric_summary(
        [(row.id, row.work, row.second.metrics if row.second else None) for row in cases]
    )
    reference = metric_summary(
        [
            (
                row.id,
                row.work,
                measure(row.original, row.vernacular, banned, entity_candidates(row.original)),
            )
            for row in examples
        ]
    )
    result = TwoPassReport(
        status=status,
        metadata=metadata,
        reference=reference,
        comparisons=comparisons,
        cases=cases,
        artifacts_path=str(artifacts_path),
        review_sample_ids=review_ids,
    )
    args.report.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
