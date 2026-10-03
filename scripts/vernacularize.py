"""Turn original chunks into conservative plan-A vernacular training inputs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import subprocess
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import jieba.posseg as pseg
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scripts.chunk_corpus import CorpusChunk, cjk_length
from scripts.rebuild_reporting import (
    AttemptRecord,
    Distribution,
    DistributionReport,
    Exclusion,
    ProviderMetadata,
    RunMetadata,
    SamplingParameters,
    append_artifacts,
    describe,
    summarize,
)
from scripts.split_corpus import CorpusSplit

PROMPT_DIR = Path(__file__).parent / "prompts"
EXCLUDED_REFERENCE_ID = "妇女生活_0015"
LOGGER = structlog.get_logger()
OUTPUT_LOCK = threading.Lock()
MIN_RATIO = 0.9
MAX_RATIO = 1.3
STRICT_COPY_SIMILARITY = 0.65
MIXED_COPY_SIMILARITY = 0.68
MAX_COPY_SIMILARITY = 0.70
HIGH_DIALOGUE_DENSITY = 0.50
MIXED_DIALOGUE_DENSITY = 0.20
HIGH_PROPER_NOUN_DENSITY = 0.10
MIXED_PROPER_NOUN_DENSITY = 0.07
PROPER_NOUN_FLAGS = {"nr", "nrfg", "nrt", "ns", "nt", "nz"}
DIALOGUE_UNIT = re.compile(r"[“”「」『』]|(?:说|问|答|喊|嚷|道)[，。！？：]")
NUMERAL_PHRASE = re.compile(
    r"(?:差不多|将近|第|初|近|约|上)?"
    r"(?:[0-9０-９]+|[零〇一二两三四五六七八九十百千万亿]+)"
    r"(?:公斤|公里|大洋|年|月|日|号|岁|个|位|名|人|口|只|条|件|家|间|所|匹|头|斤|两|克|"
    r"里|亩|州|县|国|军|代|届|次|回|遍|天|夜|块|元|角|分|束|朵|颗|粒|枚|辆|艘|顶|把|柄|"
    r"张|页|本|封|桌|场|杯|碗|盘)"
)
WIDTH_TRANSLATION = str.maketrans("０１２３４５６７８９", "0123456789")
NUMERAL_VALUE = re.compile(
    r"^(?P<modifier>差不多|将近|第|初|近|约|上)?"
    r"(?P<number>[0-9]+|[零〇一二两三四五六七八九十百千万亿]+)"
)
GRAMMATICAL_ONE = re.compile(r"^一(?:个|只|口|把|件|家|间|条|位|张|片)$")


class Generator(Protocol):
    """Injected text generation boundary."""

    def generate(self, prompt: str, **kwargs: object) -> str: ...


class Pair(BaseModel):
    """One reconstructed parallel pair."""

    id: str
    work: str
    idx: int
    vernacular: str
    original: str
    split: str


class VernacularExample(BaseModel):
    """A local-only plan-A style demonstration loaded from salvaged data."""

    id: str
    work: str = ""
    original: str
    vernacular: str


class AttemptIssue(BaseModel):
    id: str
    attempt: int
    reason: str
    length_ratio: float | None = None
    similarity: float | None = None
    similarity_limit: float | None = None
    dialogue_density: float | None = None
    proper_noun_density: float | None = None
    expected_numerals: list[str] | None = None
    actual_numerals: list[str] | None = None


class RebuildReport(BaseModel):
    status: str = "complete"
    total_requested: int
    already_done: int
    succeeded: int
    completed_after_run: int
    failed: list[str]
    out_of_range_attempts: list[AttemptIssue]
    copy_similarity_exceptions: list[AttemptIssue]
    attempt_issues: list[AttemptIssue]
    metadata: RunMetadata | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)
    per_case: list[AttemptRecord] = Field(default_factory=list)
    similarity_distribution: DistributionReport | None = None
    old_vs_new_distribution: DistributionReport | None = None
    reference_distribution: Distribution | None = None
    artifacts_path: str | None = None


PROMPT_TEMPLATE = (PROMPT_DIR / "vernacularize_a.txt").read_text(encoding="utf-8")


def numeric_phrases(text: str) -> Counter[str]:
    """Extract quantity phrases while normalizing full-width Arabic digits."""
    return Counter(
        match.group(0).translate(WIDTH_TRANSLATION) for match in NUMERAL_PHRASE.finditer(text)
    )


def numeric_facts(text: str) -> set[str]:
    """Normalize numeric values while ignoring interchangeable classifiers."""
    facts: set[str] = set()
    for phrase in numeric_phrases(text):
        if GRAMMATICAL_ONE.fullmatch(phrase):
            continue
        match = NUMERAL_VALUE.match(phrase)
        if not match:
            continue
        modifier = match.group("modifier") or ""
        modifier = {"将近": "近", "差不多": "约"}.get(modifier, modifier)
        number = match.group("number")
        if len(number) > 1 and number[0] == "一" and number[1] in "十百千万":
            number = number[1:]
        facts.add(modifier + number)
    return facts


def numerals_preserved(original: str, vernacular: str) -> bool:
    """Require every input quantity phrase without rejecting harmless new phrases."""
    return numeric_facts(original).issubset(numeric_facts(vernacular))


def build_prompt(
    original: str,
    examples: list[VernacularExample] | None = None,
    feedback: str = "",
    previous_output: str = "",
) -> str:
    demonstrations = ""
    if examples:
        example_template = (PROMPT_DIR / "example.txt").read_text(encoding="utf-8")
        rendered = [
            example_template.format(
                index=index, original=example.original, vernacular=example.vernacular
            )
            for index, example in enumerate(examples, start=1)
        ]
        demonstrations = (
            (PROMPT_DIR / "examples.txt")
            .read_text(encoding="utf-8")
            .format(examples="\n".join(rendered))
        )
    retry_instruction = ""
    if feedback:
        retry_instruction = (
            (PROMPT_DIR / "retry.txt")
            .read_text(encoding="utf-8")
            .format(feedback=feedback, previous_output=previous_output)
        )
    return demonstrations + retry_instruction + PROMPT_TEMPLATE.format(original=original)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


def paragraph_count(text: str) -> int:
    """Count non-empty paragraphs represented by separate lines."""
    return sum(bool(line.strip()) for line in text.splitlines())


def text_similarity(original: str, vernacular: str) -> float:
    """Measure direct copy similarity using the salvaged-reference metric."""
    return difflib.SequenceMatcher(None, vernacular, original).ratio()


def select_style_examples(
    examples: list[VernacularExample], *, chunk_id: str, attempt: int, seed: int
) -> list[VernacularExample]:
    """Select 2-3 demonstrations across randomly ordered work strata."""
    eligible = [example for example in examples if example.id != chunk_id]
    if not eligible:
        return []
    by_work: dict[str, list[VernacularExample]] = {}
    for example in eligible:
        by_work.setdefault(example.work or example.id.rsplit("_", 1)[0], []).append(example)
    rng = random.Random(f"{seed}:{chunk_id}:{attempt}")
    works = sorted(by_work)
    rng.shuffle(works)
    for candidates in by_work.values():
        candidates.sort(key=lambda example: example.id)
        rng.shuffle(candidates)
    count = min(2 + attempt % 2, len(eligible))
    selected: list[VernacularExample] = []
    while len(selected) < count:
        for work in works:
            if by_work[work] and len(selected) < count:
                selected.append(by_work[work].pop())
    return selected


def dialogue_density(text: str) -> float:
    """Estimate the CJK-character share occupied by dialogue-bearing units."""
    total = cjk_length(text)
    if not total:
        return 0.0
    units = re.split(r"(?<=[。！？；])|\n", text)
    dialogue_chars = sum(cjk_length(unit) for unit in units if DIALOGUE_UNIT.search(unit))
    return dialogue_chars / total


def proper_noun_density(text: str) -> float:
    """Estimate the CJK-character share tagged as names, places, or organizations."""
    total = cjk_length(text)
    if not total:
        return 0.0
    proper_chars = sum(
        cjk_length(word) for word, flag in pseg.cut(text) if flag in PROPER_NOUN_FLAGS
    )
    return proper_chars / total


def copy_similarity_limit(text: str) -> float:
    """Choose a reference-calibrated threshold from original-text densities."""
    dialogue = dialogue_density(text)
    proper_nouns = proper_noun_density(text)
    if dialogue >= HIGH_DIALOGUE_DENSITY or proper_nouns >= HIGH_PROPER_NOUN_DENSITY:
        return MAX_COPY_SIMILARITY
    if dialogue >= MIXED_DIALOGUE_DENSITY or proper_nouns >= MIXED_PROPER_NOUN_DENSITY:
        return MIXED_COPY_SIMILARITY
    return STRICT_COPY_SIMILARITY


def generate_pair(
    chunk: CorpusChunk,
    split_by_id: dict[str, str],
    generator: Generator,
    *,
    retries: int,
    seed: int,
    examples: list[VernacularExample] | None = None,
    records: list[AttemptRecord] | None = None,
    sampling: SamplingParameters | None = None,
    reference: VernacularExample | None = None,
    round_number: int = 1,
    provider_metadata: Callable[[], ProviderMetadata] | None = None,
) -> tuple[Pair | None, list[AttemptIssue]]:
    """Keep every candidate's measurements, including all rejection reasons."""
    issues: list[AttemptIssue] = []
    parameters = sampling or SamplingParameters(seed=seed)
    original_length = cjk_length(chunk.original)
    feedback = ""
    previous_output = ""
    dialogue = dialogue_density(chunk.original)
    proper_nouns = proper_noun_density(chunk.original)
    limit = copy_similarity_limit(chunk.original)
    eligible = [
        example
        for example in examples or []
        if example.id != chunk.id and example.original != chunk.original
    ]
    lowest_over_limit: float | None = None
    for attempt in range(1, retries + 2):
        selected = select_style_examples(
            eligible, chunk_id=chunk.id, attempt=attempt + round_number - 1, seed=seed
        )
        prompt = build_prompt(chunk.original, selected, feedback, previous_output)
        call_seed = seed + chunk.idx + attempt + (round_number - 1) * 100_000
        record = AttemptRecord(
            id=chunk.id,
            work=chunk.work,
            attempt=attempt,
            called_at_utc=datetime.now(UTC).isoformat(),
            seed=call_seed,
            eligible_example_ids=[example.id for example in eligible],
            selected_example_ids=[example.id for example in selected],
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            similarity_limit=limit,
            dialogue_density=dialogue,
            proper_noun_density=proper_nouns,
            original=chunk.original,
            prompt=prompt,
        )
        try:
            vernacular = _strip_fences(
                generator.generate(
                    prompt,
                    seed=call_seed,
                    temperature=parameters.temperature,
                    top_p=parameters.top_p,
                    max_tokens=parameters.max_tokens,
                )
            )
            if provider_metadata:
                record.provider = provider_metadata()
            record.output = vernacular
            record.output_sha256 = hashlib.sha256(vernacular.encode("utf-8")).hexdigest()
            record.length_ratio = (
                cjk_length(vernacular) / original_length if original_length else 0.0
            )
            record.similarity = text_similarity(chunk.original, vernacular)
            if reference:
                record.old_vs_new_similarity = text_similarity(reference.vernacular, vernacular)
            previous_output = vernacular
            record.numerical_waivers = sorted(
                {
                    phrase
                    for text in (chunk.original, vernacular)
                    for phrase in numeric_phrases(text)
                    if GRAMMATICAL_ONE.fullmatch(phrase)
                }
            )
            if record.provider.finish_reason == "length":
                record.reasons.append("provider_output_truncated")
            if not MIN_RATIO <= record.length_ratio <= MAX_RATIO:
                record.reasons.append("length_out_of_range")
            if paragraph_count(vernacular) != paragraph_count(chunk.original):
                record.reasons.append("paragraph_count_mismatch")
            if not numerals_preserved(chunk.original, vernacular):
                record.reasons.append("numeral_mismatch")
            if record.similarity > limit:
                record.reasons.append("insufficient_rewrite")
                if lowest_over_limit is None or record.similarity < lowest_over_limit:
                    lowest_over_limit = record.similarity
            record.accepted = not record.reasons
            if record.accepted:
                if records is not None:
                    records.append(record)
                return Pair(
                    id=chunk.id,
                    work=chunk.work,
                    idx=chunk.idx,
                    vernacular=vernacular,
                    original=chunk.original,
                    split=split_by_id.get(chunk.id, "train"),
                ), issues
            for reason in record.reasons:
                issues.append(
                    AttemptIssue(
                        id=chunk.id,
                        attempt=attempt,
                        reason=reason,
                        length_ratio=record.length_ratio,
                        similarity=record.similarity,
                        similarity_limit=limit,
                        dialogue_density=dialogue,
                        proper_noun_density=proper_nouns,
                        expected_numerals=sorted(numeric_facts(chunk.original))
                        if reason == "numeral_mismatch"
                        else None,
                        actual_numerals=sorted(numeric_facts(vernacular))
                        if reason == "numeral_mismatch"
                        else None,
                    )
                )
            feedback = (
                "校验问题："
                + "、".join(record.reasons)
                + f"。段落数必须为 {paragraph_count(chunk.original)}，中文字数比为 0.9–1.3。"
                + f"相似度上限 {limit:.2f}。必须保留数字短语："
                + "、".join(numeric_phrases(chunk.original))
            )
        except Exception as exc:
            # Do not include provider exception messages; they may contain credentials or text.
            record.reasons.append(type(exc).__name__)
            issues.append(AttemptIssue(id=chunk.id, attempt=attempt, reason=type(exc).__name__))
        if records is not None:
            records.append(record)
    if lowest_over_limit is not None:
        issues.append(
            AttemptIssue(
                id=chunk.id,
                attempt=retries + 1,
                reason="copy_similarity_exception",
                similarity=lowest_over_limit,
                similarity_limit=limit,
                dialogue_density=dialogue,
                proper_noun_density=proper_nouns,
            )
        )
    return None, issues


def load_chunks(path: Path) -> list[CorpusChunk]:
    with path.open(encoding="utf-8") as handle:
        return [CorpusChunk.model_validate_json(line) for line in handle if line.strip()]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {Pair.model_validate_json(line).id for line in handle if line.strip()}


def load_split(path: Path) -> dict[str, str]:
    split = CorpusSplit.model_validate_json(path.read_text(encoding="utf-8"))
    return {**dict.fromkeys(split.train, "train"), **dict.fromkeys(split.eval, "eval")}


def load_preview_reference(path: Path) -> list[CorpusChunk]:
    """Load the 13 non-anomalous salvaged originals for leave-one-out validation."""
    chunks: list[CorpusChunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("maybe_truncated", True) and row["id"] != EXCLUDED_REFERENCE_ID:
                chunks.append(
                    CorpusChunk(
                        id=row["id"], work=row["work"], idx=row["idx"], original=row["original"]
                    )
                )
    return chunks


def load_style_examples(path: Path) -> list[VernacularExample]:
    """Load short complete examples without embedding copyrighted text in source code."""
    examples: list[VernacularExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("maybe_truncated", True) and row["id"] != EXCLUDED_REFERENCE_ID:
                examples.append(
                    VernacularExample(
                        id=row["id"],
                        original=row["original"],
                        vernacular=row["vernacular"],
                        work=row["work"],
                    )
                )
    return sorted(examples, key=lambda example: cjk_length(example.original))


def stratified_sample(
    chunks: list[CorpusChunk], *, count: int, seed: int, excluded_ids: set[str] | None = None
) -> list[CorpusChunk]:
    """Sample randomly within work strata with approximately equal allocation."""
    excluded = excluded_ids or set()
    by_work: dict[str, list[CorpusChunk]] = {}
    for chunk in sorted(chunks, key=lambda item: item.id):
        if chunk.id not in excluded:
            by_work.setdefault(chunk.work, []).append(chunk)
    if count < 1 or count > sum(map(len, by_work.values())):
        raise ValueError("抽样数超出可用样本范围")
    rng = random.Random(seed)
    works = sorted(by_work)
    rng.shuffle(works)
    for candidates in by_work.values():
        rng.shuffle(candidates)
    selected: list[CorpusChunk] = []
    while len(selected) < count:
        for work in works:
            if by_work[work] and len(selected) < count:
                selected.append(by_work[work].pop())
    return sorted(selected, key=lambda chunk: chunk.id)


def select_work_balanced_fill(
    chunks: list[CorpusChunk], *, count: int, excluded_ids: set[str], seed: int = 42
) -> list[CorpusChunk]:
    """Compatibility entry point now using stratified random sampling."""
    return stratified_sample(chunks, count=count, seed=seed, excluded_ids=excluded_ids)


def run_batch(
    chunks: list[CorpusChunk],
    output: Path,
    report_path: Path,
    split_by_id: dict[str, str],
    generator: Generator,
    *,
    concurrency: int = 4,
    retries: int = 2,
    seed: int = 42,
    examples: list[VernacularExample] | None = None,
    metadata: RunMetadata | None = None,
    references: list[VernacularExample] | None = None,
    provider_metadata: Callable[[], ProviderMetadata] | None = None,
) -> RebuildReport:
    """Run a resumable batch; retain rejected outputs without filtering distributions."""
    completed = load_completed(output)
    remaining = [chunk for chunk in chunks if chunk.id not in completed]
    LOGGER.info(
        "vernacularize_start",
        total=len(chunks),
        already_done=len(completed),
        remaining=len(remaining),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed: list[str] = []
    issues: list[AttemptIssue] = []
    all_records: list[AttemptRecord] = []
    last_records: list[AttemptRecord] = []
    reference_by_id = {example.id: example for example in references or []}
    if references:
        for chunk in chunks:
            eligible = [
                example
                for example in examples or []
                if example.id != chunk.id and example.original != chunk.original
            ]
            if len(eligible) != 12:
                raise ValueError("每个留一 fold 必须恰好有 12 条候选范例")
    artifacts_path = Path("corpus/generations") / (report_path.stem + "_attempts.jsonl")
    if metadata is None:
        artifacts_path = report_path.with_suffix(".attempts.jsonl")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    initial = RebuildReport(
        status="in_progress",
        total_requested=len(chunks),
        already_done=len(completed),
        succeeded=0,
        completed_after_run=len(completed),
        failed=[],
        out_of_range_attempts=[],
        copy_similarity_exceptions=[],
        attempt_issues=[],
        metadata=metadata,
        artifacts_path=str(artifacts_path),
    )
    report_path.write_text(initial.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def process(chunk: CorpusChunk) -> tuple[Pair | None, list[AttemptIssue], list[AttemptRecord]]:
        records: list[AttemptRecord] = []
        pair, pair_issues = generate_pair(
            chunk,
            split_by_id,
            generator,
            retries=retries,
            seed=seed,
            examples=examples,
            records=records,
            sampling=metadata.sampling if metadata else None,
            reference=reference_by_id.get(chunk.id),
            round_number=metadata.round if metadata else 1,
            provider_metadata=provider_metadata,
        )
        return pair, pair_issues, records

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[
            Future[tuple[Pair | None, list[AttemptIssue], list[AttemptRecord]]], CorpusChunk
        ] = {executor.submit(process, chunk): chunk for chunk in remaining}
        for processed, future in enumerate(as_completed(futures), start=1):
            chunk = futures[future]
            pair, pair_issues, records = future.result()
            issues.extend(pair_issues)
            all_records.extend(records)
            last_records.append(records[-1])
            append_artifacts(artifacts_path, records)
            if pair is None:
                failed.append(chunk.id)
            else:
                with OUTPUT_LOCK, output.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(pair.model_dump_json() + "\n")
                succeeded += 1
            LOGGER.info(
                "vernacularize_case",
                id=chunk.id,
                similarity=records[-1].similarity,
                old_vs_new=records[-1].old_vs_new_similarity,
                reasons=records[-1].reasons,
            )
            if processed % 20 == 0 or processed == len(remaining):
                LOGGER.info(
                    "vernacularize_progress",
                    progress=f"{processed}/{len(remaining)}",
                    ok=succeeded,
                    failed=len(failed),
                )
    # Resume does not masquerade previously produced pairs as new external calls.
    if metadata:
        metadata.completed_at_utc = datetime.now(UTC).isoformat()
        metadata.exclusions.extend(
            Exclusion(id=chunk.id, reason="already_done; no_new_call_in_this_run")
            for chunk in chunks
            if chunk.id in completed
        )
    last_records.sort(key=lambda record: record.id)
    report = RebuildReport(
        total_requested=len(chunks),
        already_done=len(completed),
        succeeded=succeeded,
        completed_after_run=len(completed) + succeeded,
        failed=sorted(failed),
        out_of_range_attempts=[issue for issue in issues if issue.reason == "length_out_of_range"],
        copy_similarity_exceptions=[
            issue for issue in issues if issue.reason == "copy_similarity_exception"
        ],
        attempt_issues=sorted(issues, key=lambda issue: (issue.id, issue.attempt, issue.reason)),
        metadata=metadata,
        attempts=sorted(all_records, key=lambda record: (record.id, record.attempt)),
        per_case=last_records,
        similarity_distribution=summarize(
            last_records, expected_ids=[chunk.id for chunk in chunks]
        ),
        old_vs_new_distribution=summarize(
            last_records, compare_old=True, expected_ids=[chunk.id for chunk in chunks]
        )
        if references
        else None,
        reference_distribution=describe(
            [text_similarity(example.original, example.vernacular) for example in references]
        )
        if references
        else None,
        artifacts_path=str(artifacts_path),
    )
    report_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_metadata(
    args: argparse.Namespace,
    chunks: list[CorpusChunk],
    examples: list[VernacularExample],
) -> RunMetadata:
    """Require committed prompt code and record source hashes before any API request."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
    paths = [str(path.relative_to(Path.cwd())) for path in prompt_files]
    dirty = subprocess.run(
        [
            "git",
            "diff",
            "HEAD",
            "--",
            *paths,
            "scripts/vernacularize.py",
            "scripts/rebuild_reporting.py",
            "infra/openai_generator.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    if dirty:
        raise ValueError("外部运行前必须提交 prompt 与调用代码")
    tracked = subprocess.run(
        ["git", "ls-files", "--", *paths], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if len(tracked) != len(paths):
        raise ValueError("所有 prompt 模板必须已提交")
    exclusions: list[Exclusion] = []
    if args.style_reference:
        with args.style_reference.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("maybe_truncated", True) or row["id"] == EXCLUDED_REFERENCE_ID:
                    ratio = cjk_length(row["vernacular"]) / max(cjk_length(row["original"]), 1)
                    exclusions.append(
                        Exclusion(
                            id=row["id"],
                            reason="original_side_truncated_anomaly"
                            if row["id"] == EXCLUDED_REFERENCE_ID
                            else "source_log_marked_truncated",
                            length_ratio=ratio,
                        )
                    )
    sources = [args.chunks, args.split]
    if args.style_reference:
        sources.append(args.style_reference)
    return RunMetadata(
        model=os.environ.get("VERNACULARIZE_MODEL", "deepseek-chat"),
        endpoint_host=urlsplit(os.environ.get("LLM_BASE_URL", "")).hostname,
        sampling=SamplingParameters(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        prompt_git_commit=commit,
        prompt_path="scripts/prompts/vernacularize_a.txt",
        prompt_sha256=hashlib.sha256(
            b"".join(path.read_bytes() for path in prompt_files)
        ).hexdigest(),
        mode="leave_one_out"
        if args.preview_reference
        else "stratified_preview"
        if args.limit
        else "full_rebuild",
        round=args.round,
        retries=args.retries,
        concurrency=args.concurrency,
        sampling_method="all_13_leave_one_out"
        if args.preview_reference
        else "equal_work_allocation_random_within_work"
        if args.limit
        else "all_chunks",
        selected_ids=[chunk.id for chunk in chunks],
        example_pool_ids=[example.id for example in examples],
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        exclusions=exclusions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 OpenAI 兼容接口把原文降级改写成现代白话")
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--report", type=Path, default=Path("corpus/rebuild_report.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-reference", type=Path)
    parser.add_argument("--style-reference", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", type=int, choices=[1, 2], default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--allow-full", action="store_true", help="仅在人工明确放行全量之后使用")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.limit and not args.preview_reference and not args.allow_full:
        LOGGER.error("full_run_requires_explicit_release")
        return 1
    if args.retries < 0:
        raise ValueError("retries 不得为负数")
    load_dotenv()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        LOGGER.error("missing_environment", variable="LLM_API_KEY")
        return 1

    from infra.openai_generator import OpenAICompatibleGenerator

    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    chunks = load_chunks(args.chunks)
    examples = load_style_examples(args.style_reference) if args.style_reference else []
    references: list[VernacularExample] | None = None
    if args.preview_reference:
        references = load_style_examples(args.preview_reference)
        if len(references) != 13 or len(examples) != 13:
            raise ValueError("留一法必须使用剔除异常后的 13 条完整参照")
        if {row.id for row in references} != {row.id for row in examples}:
            raise ValueError("留一法的范例池必须正好为同一组 13 条")
        chunks = load_preview_reference(args.preview_reference)
    elif args.limit is not None:
        excluded_ids = {example.id for example in examples} | {EXCLUDED_REFERENCE_ID}
        reference_texts = {example.original for example in examples}
        candidates = [chunk for chunk in chunks if chunk.original not in reference_texts]
        chunks = stratified_sample(
            candidates, count=args.limit, seed=args.seed, excluded_ids=excluded_ids
        )
    metadata = run_metadata(args, chunks, examples)
    if args.limit and not args.preview_reference:
        selected_ids = {chunk.id for chunk in chunks}
        for chunk in load_chunks(args.chunks):
            if chunk.id not in selected_ids:
                reason = "stratified_random_not_selected"
                if chunk.id in excluded_ids or chunk.original in reference_texts:
                    reason = "few_shot_or_anomalous_reference_excluded_from_preview"
                metadata.exclusions.append(Exclusion(id=chunk.id, reason=reason))

    generator = OpenAICompatibleGenerator(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("VERNACULARIZE_MODEL", "deepseek-chat"),
    )
    report = run_batch(
        chunks,
        args.output,
        args.report,
        load_split(args.split),
        generator,
        concurrency=args.concurrency,
        retries=args.retries,
        seed=args.seed,
        examples=examples,
        metadata=metadata,
        references=references,
        provider_metadata=generator.response_metadata,
    )
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
