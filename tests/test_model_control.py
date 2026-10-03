"""CPU-only checks for fatal model routing and exact historical replay."""

import hashlib
from pathlib import Path

import pytest

import infra.openai_generator as provider
from scripts.chunk_corpus import CorpusChunk
from scripts.model_control import verify_control
from scripts.rebuild_reporting import (
    AttemptArtifact,
    AttemptRecord,
    ModelIdentityMismatchError,
    ProviderMetadata,
    RunMetadata,
    SamplingParameters,
)
from scripts.vernacularize import (
    RebuildReport,
    VernacularExample,
    build_prompt,
    run_batch,
    select_style_examples,
)
from tests.fakes import FakeMismatchGenerator, FakeOpenAI, FakeProviderCall


def test_provider_model_mismatch_is_fatal_and_blocks_further_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[FakeProviderCall] = []
    client = FakeOpenAI("deepseek-flash", requests)
    monkeypatch.setattr(provider, "OpenAI", lambda **kwargs: client)
    generator = provider.OpenAICompatibleGenerator(
        api_key="fake", base_url=None, model="deepseek-chat"
    )
    for _ in range(2):
        with pytest.raises(ModelIdentityMismatchError, match="deepseek-flash"):
            generator.generate("虚构内容", seed=42)
    assert len(requests) == 1


def test_exact_model_and_all_sampling_parameters_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[FakeProviderCall] = []
    client = FakeOpenAI("deepseek-v4-pro", requests)
    monkeypatch.setattr(provider, "OpenAI", lambda **kwargs: client)
    generator = provider.OpenAICompatibleGenerator(
        api_key="fake", base_url=None, model="deepseek-v4-pro"
    )
    assert (
        generator.generate(
            "虚构",
            seed=100123,
            temperature=0.2,
            top_p=1.0,
            max_tokens=2048,
            thinking_mode="disabled",
        )
        == "假数据"
    )
    payload = requests[0].model_dump()
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["seed"] == 100123
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 2048
    assert payload["extra_body"] == {"thinking": {"type": "disabled"}}
    assert generator.response_metadata().model == "deepseek-v4-pro"


def test_batch_mismatch_does_not_retry_and_records_every_cancelled_case(tmp_path: Path) -> None:
    chunks = [
        CorpusChunk(id=f"虚构_{idx:04d}", work="虚构", idx=idx, original="甲" * 210)
        for idx in range(6)
    ]
    fake = FakeMismatchGenerator()
    result = run_batch(
        chunks,
        tmp_path / "pairs.jsonl",
        tmp_path / "report.json",
        {},
        fake,
        concurrency=1,
        retries=3,
        provider_metadata=fake.response_metadata,
    )
    assert len(fake.calls) == 1
    assert result.status == "aborted_model_mismatch"
    assert result.failed == [chunk.id for chunk in chunks]
    assert len(result.attempt_issues) == 6
    assert result.per_case[0].provider.model == "returned"
    assert result.per_case[0].reasons == ["model_identity_mismatch"]
    assert result.similarity_distribution is not None
    assert result.similarity_distribution.mean_interval_status == "incomplete"
    assert not (tmp_path / "pairs.jsonl").exists()


@pytest.fixture
def control_data() -> tuple[
    RebuildReport,
    list[AttemptArtifact],
    list[CorpusChunk],
    list[VernacularExample],
    SamplingParameters,
]:
    chunks = [
        CorpusChunk(id=f"虚构_{idx:04d}", work="虚构", idx=idx, original=f"虚构段落{idx}")
        for idx in range(20)
    ]
    examples = [
        VernacularExample(
            id=f"范例_{idx:04d}", work="范例", original=f"虚构范例{idx}", vernacular="改写的假数据"
        )
        for idx in range(13)
    ]
    sampling = SamplingParameters(thinking_mode="disabled")
    records: list[AttemptRecord] = []
    artifacts: list[AttemptArtifact] = []
    for chunk in chunks:
        selected = select_style_examples(examples, chunk_id=chunk.id, attempt=2, seed=42)
        prompt = build_prompt(chunk.original, selected)
        record = AttemptRecord(
            id=chunk.id,
            work=chunk.work,
            attempt=1,
            called_at_utc="fake-date",
            seed=100043 + chunk.idx,
            eligible_example_ids=[row.id for row in examples],
            selected_example_ids=[row.id for row in selected],
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            provider=ProviderMetadata(model="deepseek-flash"),
            similarity_limit=0.65,
            dialogue_density=0.0,
            proper_noun_density=0.0,
        )
        records.append(record)
        artifacts.append(
            AttemptArtifact(record=record, original=chunk.original, prompt=prompt, output="假数据")
        )
    metadata = RunMetadata(
        model="deepseek-chat",
        sampling=SamplingParameters(),
        prompt_git_commit="fake-commit",
        prompt_path="scripts/prompts/vernacularize_a.txt",
        prompt_sha256="fake-hash",
        mode="stratified_preview",
        round=2,
        retries=0,
        concurrency=4,
        sampling_method="stratified",
        selected_ids=[row.id for row in chunks],
        example_pool_ids=[row.id for row in examples],
        source_sha256={},
        exclusions=[],
    )
    report = RebuildReport(
        total_requested=20,
        already_done=0,
        succeeded=0,
        completed_after_run=0,
        failed=metadata.selected_ids,
        out_of_range_attempts=[],
        copy_similarity_exceptions=[],
        attempt_issues=[],
        metadata=metadata,
        attempts=records,
        per_case=records,
    )
    return report, artifacts, chunks, examples, sampling


def test_control_accepts_exact_replay(
    control_data: tuple[
        RebuildReport,
        list[AttemptArtifact],
        list[CorpusChunk],
        list[VernacularExample],
        SamplingParameters,
    ],
) -> None:
    verify_control(*control_data)


@pytest.mark.parametrize("change", ["prompt", "seed", "case", "temperature", "examples"])
def test_control_rejects_any_other_changed_variable(
    control_data: tuple[
        RebuildReport,
        list[AttemptArtifact],
        list[CorpusChunk],
        list[VernacularExample],
        SamplingParameters,
    ],
    change: str,
) -> None:
    report, artifacts, chunks, examples, sampling = control_data
    if change == "prompt":
        artifacts[0].prompt += "额外指令"
    elif change == "seed":
        report.per_case[0].seed += 1
    elif change == "case":
        chunks.reverse()
    elif change == "temperature":
        sampling.temperature = 0.9
    else:
        examples.reverse()
    with pytest.raises(ValueError):
        verify_control(report, artifacts, chunks, examples, sampling)
