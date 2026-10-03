"""Replay round2 unchanged on the canonical Pro model; never launch a full batch."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import structlog
from dotenv import load_dotenv

from scripts.chunk_corpus import CorpusChunk
from scripts.rebuild_reporting import AttemptArtifact, SamplingParameters
from scripts.vernacularize import (
    PROMPT_DIR,
    RebuildReport,
    VernacularExample,
    build_prompt,
    load_chunks,
    load_split,
    load_style_examples,
    run_batch,
    select_style_examples,
)

LOGGER = structlog.get_logger()
CONTROL_MODEL = "deepseek-v4-pro"


def verify_control(
    baseline: RebuildReport,
    artifacts: list[AttemptArtifact],
    chunks: list[CorpusChunk],
    examples: list[VernacularExample],
    sampling: SamplingParameters,
) -> None:
    """Reject any change in cases, expanded prompts, examples, or call parameters."""
    metadata = baseline.metadata
    if (
        metadata is None
        or metadata.round != 2
        or metadata.retries != 0
        or baseline.status != "complete"
    ):
        raise ValueError("Control requires the unretried round2 baseline")
    if metadata.model != "deepseek-chat" or metadata.sampling.thinking_mode is not None:
        raise ValueError("Historical non-thinking mode inference only applies to deepseek-chat")
    if (
        metadata.sampling.model_dump(exclude={"thinking_mode"})
        != sampling.model_dump(exclude={"thinking_mode"})
        or sampling.thinking_mode != "disabled"
    ):
        raise ValueError("Sampling parameters changed")
    ids = metadata.selected_ids
    if len(ids) != 20 or len(set(ids)) != 20 or [chunk.id for chunk in chunks] != ids:
        raise ValueError("Control must preserve all 20 baseline case IDs and their order")
    if len(examples) != 13 or [example.id for example in examples] != metadata.example_pool_ids:
        raise ValueError("Example pool changed")
    records = {record.id: record for record in baseline.per_case}
    archived = {artifact.record.id: artifact for artifact in artifacts}
    if len(baseline.attempts) != 20 or len(records) != 20 or len(artifacts) != 20:
        raise ValueError("Baseline must retain exactly 20 single-call observations")
    if set(records) != set(ids) or set(archived) != set(ids):
        raise ValueError("Missing baseline observations or archives")
    for chunk in chunks:
        record, artifact = records[chunk.id], archived[chunk.id]
        eligible = [
            row for row in examples if row.id != chunk.id and row.original != chunk.original
        ]
        selected = select_style_examples(
            eligible, chunk_id=chunk.id, attempt=metadata.round, seed=sampling.seed
        )
        prompt = build_prompt(chunk.original, selected)
        call_seed = sampling.seed + chunk.idx + 1 + (metadata.round - 1) * 100_000
        if (
            record.attempt != 1
            or record.provider.model != "deepseek-flash"
            or record.seed != call_seed
            or record.eligible_example_ids != [row.id for row in eligible]
            or record.selected_example_ids != [row.id for row in selected]
            or record.prompt_sha256 != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or artifact.original != chunk.original
            or artifact.prompt != prompt
            or artifact.record.model_dump() != record.model_dump()
        ):
            raise ValueError(f"Baseline replay differs for {chunk.id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="单变量模型对照：原样重放 round2 的 20 条")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    if args.output.exists() or args.report.exists():
        raise ValueError("Control needs fresh paths, not resume or overwrite")
    if (Path("corpus/generations") / (args.report.stem + "_attempts.jsonl")).exists():
        raise ValueError("Control artifact archive already exists")
    baseline = RebuildReport.model_validate_json(args.baseline.read_text(encoding="utf-8"))
    if baseline.metadata is None or baseline.artifacts_path is None:
        raise ValueError("Baseline provenance is incomplete")
    metadata = baseline.metadata.model_copy(deep=True)
    for source, digest in metadata.source_sha256.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed: {source}")
    prompt_digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(PROMPT_DIR.glob("*.txt")))
    ).hexdigest()
    if prompt_digest != metadata.prompt_sha256:
        raise ValueError("Prompt templates changed")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts", "infra"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("Commit the control implementation before external calls")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    all_chunks = {chunk.id: chunk for chunk in load_chunks(Path("corpus/chunks.jsonl"))}
    chunks = [all_chunks[case_id] for case_id in metadata.selected_ids]
    examples = load_style_examples(Path("corpus/salvaged_pairs.jsonl"))
    artifacts_path = Path(baseline.artifacts_path)
    artifacts = [
        AttemptArtifact.model_validate_json(line)
        for line in artifacts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sampling = metadata.sampling.model_copy(update={"thinking_mode": "disabled"})
    verify_control(baseline, artifacts, chunks, examples, sampling)
    load_dotenv()
    endpoint = os.environ.get("LLM_BASE_URL", "")
    if urlsplit(endpoint).hostname != "api.deepseek.com":
        raise ValueError("Authorized endpoint must be api.deepseek.com")
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is missing")
    metadata.model = CONTROL_MODEL
    metadata.sampling = sampling
    metadata.started_at_utc = datetime.now(UTC).isoformat()
    metadata.completed_at_utc = None
    metadata.code_git_commit = current_commit
    metadata.endpoint_host = urlsplit(endpoint).hostname
    metadata.mode = "model_control_replay"
    metadata.sampling_method = "exact_replay_of_stratified_round2"
    metadata.control_baseline = str(args.baseline)
    metadata.control_baseline_sha256 = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    metadata.control_verification = (
        "20 expanded prompts, originals, example IDs, per-call seeds and sampling verified; "
        "historical deepseek-chat non-thinking mode inferred from official alias documentation; "
        "new Pro request explicitly disables thinking. No immutable backend snapshot or "
        "provider acknowledgement of seed is available."
    )
    from infra.openai_generator import OpenAICompatibleGenerator

    generator = OpenAICompatibleGenerator(api_key=api_key, base_url=endpoint, model=CONTROL_MODEL)
    LOGGER.info(
        "model_control_preflight", cases=20, model=CONTROL_MODEL, baseline=str(args.baseline)
    )
    report = run_batch(
        chunks,
        args.output,
        args.report,
        load_split(Path("corpus/split.json")),
        generator,
        concurrency=metadata.concurrency,
        retries=metadata.retries,
        seed=sampling.seed,
        examples=examples,
        metadata=metadata,
        provider_metadata=generator.response_metadata,
    )
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
