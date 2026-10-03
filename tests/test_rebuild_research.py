from collections import Counter
from pathlib import Path

import pytest

from scripts.chunk_corpus import CorpusChunk
from scripts.rebuild_reporting import describe
from scripts.vernacularize import VernacularExample, run_batch, stratified_sample
from tests.fakes import FakeGenerator


def test_stratified_random_sample_is_order_invariant_and_covers_works() -> None:
    chunks = [
        CorpusChunk(id=f"{work}_{index:04d}", work=work, idx=index, original="虚构场景")
        for work in "甲乙丙丁戊己"
        for index in range(30)
    ]
    selected = stratified_sample(chunks, count=20, seed=42, excluded_ids={"甲_0000"})
    repeated = stratified_sample(
        list(reversed(chunks)), count=20, seed=42, excluded_ids={"甲_0000"}
    )
    assert selected == repeated
    assert set(Counter(chunk.work for chunk in selected).values()) == {3, 4}
    assert len({chunk.work for chunk in selected}) == 6
    assert "甲_0000" not in {chunk.id for chunk in selected}
    assert selected != stratified_sample(chunks, count=20, seed=43)


def test_strict_leave_one_out_records_all_12_eligible_examples(tmp_path: Path) -> None:
    references = [
        VernacularExample(
            id=f"虚构_{index:04d}",
            work="虚构",
            original="甲" * (210 + index),
            vernacular="乙" * (210 + index),
        )
        for index in range(13)
    ]
    chunks = [
        CorpusChunk(id=row.id, work="虚构", idx=index, original=row.original)
        for index, row in enumerate(references)
    ]
    fake = FakeGenerator()
    result = run_batch(
        chunks,
        tmp_path / "pairs.jsonl",
        tmp_path / "report.json",
        {},
        fake,
        examples=references,
        references=references,
        retries=0,
    )
    assert len(fake.calls) == 13
    assert len(result.per_case) == 13
    for record in result.per_case:
        assert len(record.eligible_example_ids) == 12
        assert record.id not in record.eligible_example_ids
        assert record.id not in record.selected_example_ids
        assert set(record.selected_example_ids) <= set(record.eligible_example_ids)
        assert record.old_vs_new_similarity is not None
    assert result.old_vs_new_distribution is not None
    assert result.old_vs_new_distribution.overall.count == 13


def test_invalid_leave_one_pool_fails_before_model_calls(tmp_path: Path) -> None:
    fake = FakeGenerator()
    row = VernacularExample(id="虚构_0001", original="甲" * 210, vernacular="乙" * 210)
    chunk = CorpusChunk(id=row.id, work="虚构", idx=1, original=row.original)
    with pytest.raises(ValueError, match="12"):
        run_batch(
            [chunk],
            tmp_path / "pairs.jsonl",
            tmp_path / "report.json",
            {},
            fake,
            examples=[row],
            references=[row],
            retries=0,
        )
    assert not fake.calls


def test_failed_candidates_remain_in_distribution_and_local_archive(tmp_path: Path) -> None:
    chunk = CorpusChunk(id="虚构_0001", work="虚构", idx=1, original="河边" * 110)
    result = run_batch(
        [chunk], tmp_path / "pairs.jsonl", tmp_path / "report.json", {}, FakeGenerator(), retries=0
    )
    assert result.failed == [chunk.id]
    assert result.similarity_distribution is not None
    assert result.similarity_distribution.overall.mean == 1.0
    assert result.per_case[0].similarity == 1.0
    assert "insufficient_rewrite" in result.per_case[0].reasons
    assert not (tmp_path / "pairs.jsonl").exists()
    assert "河边" in (tmp_path / "report.attempts.jsonl").read_text(encoding="utf-8")
    assert "河边" not in (tmp_path / "report.json").read_text(encoding="utf-8")


def test_distribution_has_defined_standard_deviations_and_quartiles() -> None:
    summary = describe([0.2, 0.4, 0.6, 0.8])
    assert summary.mean == pytest.approx(0.5)
    assert summary.std == pytest.approx(0.25819888974)
    assert summary.population_std == pytest.approx(0.22360679775)
    assert summary.q1 == pytest.approx(0.35)
    assert summary.q3 == pytest.approx(0.65)
