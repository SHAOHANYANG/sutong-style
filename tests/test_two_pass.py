"""Synthetic-only tests for the frozen T0.0 experiment, never calling a service."""

import json
from pathlib import Path

import pytest

from scripts.chunk_corpus import CorpusChunk
from scripts.extract_replacements import ReplacementLexicon, classify_term, extract_lexicon
from scripts.extract_replacements import main as export_replacements
from scripts.rebuild_metrics import final_failures, measure, numeric_values, pinc, source_bleu
from scripts.rebuild_reporting import AttemptRecord, RunMetadata, SamplingParameters
from scripts.two_pass_preview import StageArtifact, metric_summary, run_experiment
from scripts.vernacularize import RebuildReport, VernacularExample
from tests.fakes import FakeMismatchGenerator, FakeTwoPassGenerator


def test_committed_lexicon_encoding_and_synthetic_reproducibility() -> None:
    lexicon = ReplacementLexicon.model_validate_json(
        Path("scripts/data/mandatory_replacements.json").read_text(encoding="utf-8")
    )
    assert len(lexicon.terms) == 96
    assert all("\ufffd" not in path for path in lexicon.source_sha256)
    assert len(lexicon.source_sha256) == 3
    assert {row.term for row in lexicon.terms} >= {"仿佛", "绯红"}


def test_replacement_export_is_ascii_json_with_lossless_chinese(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "corpus" / "raw"
    raw_path.mkdir(parents=True)
    (raw_path / "自编样例.txt").write_text("蓦然绯红。蓦然绯红。", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    export_replacements()
    exported = capsys.readouterr().out
    assert exported.isascii()
    for encoding in ("utf-8", "cp936"):
        transported = exported.encode(encoding).decode(encoding)
        lexicon = ReplacementLexicon.model_validate(json.loads(transported)["lexicon"])
        assert {row.term for row in lexicon.terms} >= {"蓦然", "绯红"}
        assert list(lexicon.source_sha256) == [str(Path("corpus/raw/自编样例.txt"))]


def test_pinc_candidate_direction_repetitions_and_preprocessing() -> None:
    assert pinc("甲乙", "甲甲丙丙丙", 1) == pytest.approx(0.6)
    assert pinc("甲甲丙丙丙", "甲乙", 1) == pytest.approx(0.5)
    assert pinc("甲，乙！丙", "甲 乙3丙。", 2) == 0
    assert pinc("甲乙", "甲", 2) is None
    assert pinc("甲", "丙乙", 2) == 1
    with pytest.raises(ValueError):
        pinc("甲", "乙", 0)
    assert source_bleu("甲乙", "甲，乙") == pytest.approx(100)
    assert source_bleu("甲乙", "") == 0


def test_values_not_forms_and_entity_numeric_protection() -> None:
    entities = ["四姨太"]
    old = numeric_values("四姨太三月初九带着三百元去了十七州八十县。", entities)
    new = numeric_values("四姨太3月9号带着300元去了17个州80个县。", entities)
    assert old.values == new.values == {"3", "9", "300", "17", "80"}
    assert not new.chinese_forms
    assert numeric_values("万名将士").values == {"10000"}
    assert numeric_values("一个人").grammatical_waivers


def test_pinc_gates_repair_delta_and_extra_facts_visible() -> None:
    first = measure("甲乙丙丁戊己庚辛", "壬癸子丑寅卯辰巳", [], [])
    final = first.model_copy(update={"pinc4": 0.74, "pinc6": 0.84, "added_values": ["2"]})
    failures = final_failures(first, final)
    assert "pinc4_below_limit" in failures
    assert "pinc6_below_limit" in failures
    assert "pinc4_repair_decrease" in failures
    assert "numeral_added" in failures
    assert first.pinc_mean_1_to_4 == 1
    assert metric_summary([("甲", "虚构", None)]).missing_ids == ["甲"]


def test_replacement_extraction_is_fixed_and_filters_colloquial_l(tmp_path: Path) -> None:
    assert not classify_term("怎么办", "l")
    assert not classify_term("怎么回事", "l")
    assert not classify_term("四姨太", "nr")
    assert "literary_word" in classify_term("蓦然", "d")
    path = tmp_path / "fiction.txt"
    path.write_text("蓦然绯红。蓦然绯红。", encoding="utf-8")
    first, second = extract_lexicon([path]), extract_lexicon([path])
    assert first == second
    assert all(row.count >= 2 for row in first.terms)
    assert {row.term for row in first.terms} >= {"蓦然", "绯红"}


def test_manual_replacement_catalog_has_approved_67_terms() -> None:
    catalog = json.loads(
        Path("scripts/data/approved_replacement_terms.json").read_text(encoding="utf-8")
    )
    assert catalog["version"] == "manual-approved-v1"
    assert len(catalog["terms"]) == len(set(catalog["terms"])) == 67
    assert {"仿佛", "犹如"} <= set(catalog["terms"])
    assert not {"莫名其妙", "小心翼翼", "不可思议"} & set(catalog["terms"])


def test_destroy_prompt_carries_category_rules_and_all_reviewed_examples() -> None:
    prompt = Path("scripts/prompts/two_pass/destroy.txt").read_text(encoding="utf-8")
    catalog = json.loads(
        Path("scripts/data/approved_replacement_terms.json").read_text(encoding="utf-8")
    )
    assert "清单只是示例，不是穷举" in prompt
    assert "莫名其妙、小心翼翼、不可思议是口语常用例外" in prompt
    assert "之、其、乃、遂、颇、甚、矣" in prompt
    assert "凝视、伫立、颔首、蹙眉、噙、沁" in prompt
    assert "仿佛、犹如、宛若" in prompt
    assert "哭/泣" in prompt and "之间轮换" in prompt
    assert "带序数的称谓" in prompt
    assert "大少爷、大太太、二/三/四太太、二/三/四姨太" in prompt
    assert "不得为了降重合率删掉事实" in prompt
    assert all(f"{term} →" in prompt for term in catalog["terms"])


def test_repair_prompt_only_restores_names_and_numbers() -> None:
    prompt = Path("scripts/prompts/two_pass/repair.txt").read_text(encoding="utf-8")
    assert "只允许做两件事" in prompt
    assert "不得恢复任何被有意替换的形容词、副词、成语、四字格" in prompt
    assert "不得以“更贴近原文”为由" in prompt
    assert "PINC-4和PINC-6不得低于第一遍" in prompt
    assert "entity 或 numeral" in prompt
    assert "event/causality/dialogue/other" not in prompt


def test_repair_pinc_decrease_tolerance_is_zero() -> None:
    first = measure("甲乙丙丁戊己庚辛壬癸", "子丑寅卯辰巳午未申酉", [], [])
    assert first.pinc4 == 1 and first.pinc6 == 1
    slight = first.model_copy(update={"pinc4": 0.99, "pinc6": 0.99})
    failures = final_failures(first, slight)
    assert "pinc4_repair_decrease" in failures
    assert "pinc6_repair_decrease" in failures
    assert "pinc4_below_limit" not in failures
    assert "pinc4_repair_decrease" not in final_failures(first, first)


def fixture_data() -> tuple[list[CorpusChunk], RebuildReport, list[VernacularExample], RunMetadata]:
    chunks = [
        CorpusChunk(id=f"虚构_{i:04d}", work="虚构", idx=i, original="甲乙在河边说话" * 32)
        for i in range(20)
    ]
    example = VernacularExample(
        id="范例_0000", work="范例", original="自己编的例子", vernacular="假白话"
    )
    records = [
        AttemptRecord(
            id=row.id,
            work=row.work,
            attempt=1,
            called_at_utc="fake-date",
            seed=100043 + i,
            eligible_example_ids=[example.id],
            selected_example_ids=[example.id],
            prompt_sha256="fake-hash",
            similarity_limit=0.65,
            dialogue_density=0,
            proper_noun_density=0,
        )
        for i, row in enumerate(chunks)
    ]
    metadata = RunMetadata(
        model="deepseek-v4-pro",
        sampling=SamplingParameters(thinking_mode="disabled"),
        prompt_git_commit="fake-commit",
        prompt_path="fake",
        prompt_sha256="fake-hash",
        mode="test",
        round=1,
        retries=0,
        concurrency=1,
        sampling_method="stratified",
        selected_ids=[row.id for row in chunks],
        example_pool_ids=[example.id],
        source_sha256={},
        exclusions=[],
    )
    baseline = RebuildReport(
        total_requested=20,
        already_done=0,
        succeeded=0,
        completed_after_run=0,
        failed=[],
        out_of_range_attempts=[],
        copy_similarity_exceptions=[],
        attempt_issues=[],
        per_case=records,
    )
    return chunks, baseline, [example], metadata


@pytest.mark.parametrize("invalid_json", [False, True])
def test_two_calls_per_case_seeds_preserved_no_retries(tmp_path: Path, invalid_json: bool) -> None:
    chunks, baseline, examples, metadata = fixture_data()
    fake = FakeTwoPassGenerator(invalid_json=invalid_json)
    artifact_path = tmp_path / "stages.jsonl"
    cases, status = run_experiment(
        chunks,
        baseline,
        examples,
        [],
        fake,
        metadata,
        tmp_path / "report.json",
        artifact_path,
    )
    assert status == "complete"
    assert len(cases) == 20 and len(fake.calls) == 40
    for i in range(20):
        assert fake.calls[2 * i].seed == fake.calls[2 * i + 1].seed == 100043 + i
        assert fake.calls[2 * i].temperature == 0.2
    artifacts = [
        StageArtifact.model_validate_json(line)
        for line in artifact_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(artifacts) == 40
    if invalid_json:
        assert all("repair_json_invalid" in row.reasons for row in cases)
        assert all(row.second and row.second.metrics is None for row in cases)
    else:
        assert all(
            row.first and row.first.metrics and row.second and row.second.metrics for row in cases
        )


def test_mismatch_stops_queued_calls_and_preserves_failures(tmp_path: Path) -> None:
    chunks, baseline, examples, metadata = fixture_data()
    fake = FakeMismatchGenerator()
    cases, status = run_experiment(
        chunks,
        baseline,
        examples,
        [],
        fake,
        metadata,
        tmp_path / "report.json",
        tmp_path / "stages.jsonl",
    )
    assert status == "aborted_model_mismatch"
    assert len(fake.calls) == 1
    assert len(cases) == 20
    assert all(not row.accepted_by_deterministic_gates and row.reasons for row in cases)
