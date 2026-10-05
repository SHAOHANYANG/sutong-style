"""Typed provenance and descriptive statistics for corpus reconstruction."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field


class SamplingParameters(BaseModel):
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: int = 42
    thinking_mode: str | None = None


class ModelIdentityMismatchError(RuntimeError):
    """Fatal provider routing error; never retry it as a bad generation."""

    def __init__(self, requested: str, returned: str) -> None:
        self.requested = requested
        self.returned = returned
        super().__init__(f"Requested model {requested!r}, received {returned!r}")


class ProviderMetadata(BaseModel):
    model: str | None = None
    response_id: str | None = None
    created: int | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None


class Exclusion(BaseModel):
    id: str
    reason: str
    length_ratio: float | None = None


class RunMetadata(BaseModel):
    model: str
    endpoint_host: str | None = None
    started_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at_utc: str | None = None
    sampling: SamplingParameters
    prompt_git_commit: str
    code_git_commit: str | None = None
    control_baseline: str | None = None
    control_baseline_sha256: str | None = None
    control_verification: str | None = None
    prompt_path: str
    prompt_sha256: str
    mode: str
    round: int
    retries: int
    concurrency: int
    sampling_method: str
    selected_ids: list[str]
    example_pool_ids: list[str]
    source_sha256: dict[str, str]
    exclusions: list[Exclusion]
    gate: str = "historical"
    snapshot_limitation: str = "Hosted alias may not expose an immutable model snapshot."


class AttemptRecord(BaseModel):
    id: str
    work: str
    attempt: int
    called_at_utc: str
    seed: int
    eligible_example_ids: list[str]
    selected_example_ids: list[str]
    prompt_sha256: str
    output_sha256: str | None = None
    provider: ProviderMetadata = Field(default_factory=ProviderMetadata)
    similarity: float | None = None
    old_vs_new_similarity: float | None = None
    similarity_limit: float
    dialogue_density: float
    proper_noun_density: float
    length_ratio: float | None = None
    accepted: bool = False
    reasons: list[str] = Field(default_factory=list)
    pinc1: float | None = None
    pinc2: float | None = None
    pinc3: float | None = None
    pinc4: float | None = None
    pinc6: float | None = None
    sbleu: float | None = None
    observed_name_gaps: list[str] = Field(default_factory=list)
    numerical_waivers: list[str] = Field(default_factory=list)
    numerical_waiver_reason: str = "grammatical_one_classifier_exemption_per_QUESTIONS_Q6"
    original: str = Field(default="", exclude=True)
    prompt: str = Field(default="", exclude=True)
    output: str | None = Field(default=None, exclude=True)


class AttemptArtifact(BaseModel):
    record: AttemptRecord
    original: str
    prompt: str
    output: str | None


class Distribution(BaseModel):
    count: int
    mean: float | None = None
    std: float | None = None
    population_std: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    minimum: float | None = None
    maximum: float | None = None


class DistributionReport(BaseModel):
    expected_count: int
    missing_ids: list[str]
    overall: Distribution
    by_work: dict[str, Distribution]
    mean_interval_status: str


def describe(values: list[float]) -> Distribution:
    """Report population and sample SD, with linearly interpolated quartiles."""
    if not values:
        return Distribution(count=0)
    return Distribution(
        count=len(values),
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else None,
        population_std=statistics.pstdev(values),
        median=statistics.median(values),
        q1=float(np.quantile(values, 0.25, method="linear")),
        q3=float(np.quantile(values, 0.75, method="linear")),
        minimum=min(values),
        maximum=max(values),
    )


def summarize(
    records: list[AttemptRecord],
    *,
    compare_old: bool = False,
    expected_ids: list[str] | None = None,
) -> DistributionReport:
    values: list[float] = []
    by_work: dict[str, list[float]] = {}
    expected = expected_ids if expected_ids is not None else [record.id for record in records]
    present = {record.id for record in records}
    missing: list[str] = [case_id for case_id in expected if case_id not in present]
    for record in records:
        value = record.old_vs_new_similarity if compare_old else record.similarity
        if value is None:
            missing.append(record.id)
        else:
            values.append(value)
            by_work.setdefault(record.work, []).append(value)
    overall = describe(values)
    status = "not_applicable_old_vs_new" if compare_old else "incomplete"
    if not compare_old and not missing and overall.mean is not None:
        status = (
            "below_interval"
            if overall.mean < 0.51
            else "above_interval"
            if overall.mean > 0.61
            else "within_interval"
        )
    return DistributionReport(
        expected_count=len(expected),
        missing_ids=missing,
        overall=overall,
        by_work={work: describe(group) for work, group in sorted(by_work.items())},
        mean_interval_status=status,
    )


def append_artifacts(path: Path, records: list[AttemptRecord]) -> None:
    """Archive every prompt and output locally, including rejected candidates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            artifact = AttemptArtifact(
                record=record, original=record.original, prompt=record.prompt, output=record.output
            )
            handle.write(artifact.model_dump_json() + "\n")
