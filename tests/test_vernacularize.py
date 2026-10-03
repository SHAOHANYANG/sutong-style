from pathlib import Path

from scripts.chunk_corpus import CorpusChunk
from scripts.vernacularize import (
    Pair,
    VernacularExample,
    copy_similarity_limit,
    dialogue_density,
    generate_pair,
    numerals_preserved,
    numeric_phrases,
    paragraph_count,
    proper_noun_density,
    run_batch,
    select_style_examples,
    text_similarity,
)
from tests.fakes import FakeGenerator


def test_similarity_matches_vernacular_first_reference_metric() -> None:
    # SequenceMatcher is order-sensitive; these strings expose that difference.
    assert text_similarity("tide", "diet") == 0.5


def test_few_shot_rotation_is_reproducible_and_excludes_target() -> None:
    examples = [
        VernacularExample(id=str(index), original="虚构原文", vernacular="虚构白话")
        for index in range(14)
    ]
    first = select_style_examples(examples, chunk_id="0", attempt=1, seed=42)
    second = select_style_examples(examples, chunk_id="0", attempt=2, seed=42)

    assert len(first) == 3
    assert len(second) == 2
    assert [example.id for example in first] != [example.id for example in second]
    assert all(example.id != "0" for example in first + second)
    assert first == select_style_examples(examples, chunk_id="0", attempt=1, seed=42)


def test_restart_processes_only_remaining(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    report = tmp_path / "report.json"
    chunks = [
        CorpusChunk(id=f"测试_{idx:04d}", work="测试", idx=idx, original="想不到" + "甲" * 210)
        for idx in range(1, 4)
    ]
    output.write_text(
        Pair(
            id=chunks[0].id,
            work="测试",
            idx=1,
            vernacular=chunks[0].original,
            original=chunks[0].original,
            split="train",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    fake = FakeGenerator()

    result = run_batch(chunks, output, report, {}, fake, concurrency=2)

    assert result.already_done == 1
    assert result.succeeded == 2
    assert len(fake.calls) == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_out_of_range_result_is_reported(tmp_path: Path) -> None:
    class ShortGenerator:
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "太短"

    chunks = [CorpusChunk(id="测试_0001", work="测试", idx=1, original="甲" * 210)]
    result = run_batch(
        chunks,
        tmp_path / "pairs.jsonl",
        tmp_path / "report.json",
        {},
        ShortGenerator(),
        retries=1,
    )

    assert result.failed == ["测试_0001"]
    assert len(result.out_of_range_attempts) == 2


def test_paragraph_mismatch_is_retried() -> None:
    class ParagraphGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                return "甲" * 105 + "\n" + "甲" * 105
            return "乙" * 210

    generator = ParagraphGenerator()
    chunk = CorpusChunk(id="测试_0001", work="测试", idx=1, original="甲" * 210)

    pair, issues = generate_pair(chunk, {}, generator, retries=1, seed=42)

    assert pair is not None
    assert generator.calls == 2
    assert [issue.reason for issue in issues] == ["paragraph_count_mismatch"]
    assert paragraph_count(pair.vernacular) == paragraph_count(chunk.original)


def test_copied_original_is_retried() -> None:
    class CopyingGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            return "甲" * 210 if self.calls == 1 else "乙" * 210

    generator = CopyingGenerator()
    chunk = CorpusChunk(id="测试_0001", work="测试", idx=1, original="甲" * 210)

    pair, issues = generate_pair(chunk, {}, generator, retries=1, seed=42)

    assert pair is not None
    assert generator.calls == 2
    assert [issue.reason for issue in issues] == ["insufficient_rewrite"]
    assert text_similarity(chunk.original, pair.vernacular) == 0.0


def test_copy_similarity_limit_uses_original_density_tiers() -> None:
    narrative = "他沿着河岸走了很久。" * 20
    mixed = "他说，你先回去。" * 7 + "他沿着河岸走了很久。" * 13
    dialogue = "“你先回去。”他说。" * 20

    assert copy_similarity_limit(narrative) == 0.65
    assert copy_similarity_limit(mixed) == 0.68
    assert copy_similarity_limit(dialogue) == 0.70
    assert dialogue_density(dialogue) >= 0.50
    assert proper_noun_density(narrative) < 0.10


def test_irreducible_copy_is_rejected_and_reported_for_review(tmp_path: Path) -> None:
    original = "他沿着河岸慢慢往前走，始终没有回头。" * 20

    class CopyingGenerator:
        def generate(self, prompt: str, **kwargs: object) -> str:
            return original

    chunk = CorpusChunk(id="测试_0001", work="测试", idx=1, original=original)
    output = tmp_path / "pairs.jsonl"
    result = run_batch(
        [chunk],
        output,
        tmp_path / "report.json",
        {},
        CopyingGenerator(),
        retries=1,
    )

    assert result.failed == ["测试_0001"]
    assert result.succeeded == 0
    assert len(result.copy_similarity_exceptions) == 1
    assert result.copy_similarity_exceptions[0].similarity_limit == 0.65
    assert not output.exists()


def test_changed_numeral_modifier_is_retried() -> None:
    class NumeralGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            prefix = "上万人" if self.calls == 1 else "万人"
            return prefix + "乙" * 208

    generator = NumeralGenerator()
    chunk = CorpusChunk(id="测试_0001", work="测试", idx=1, original="万人" + "甲" * 208)

    pair, issues = generate_pair(chunk, {}, generator, retries=1, seed=42)

    assert pair is not None
    assert generator.calls == 2
    assert [issue.reason for issue in issues] == ["numeral_mismatch"]


def test_numeric_phrases_normalize_full_width_digits() -> None:
    assert numeric_phrases("１９３８年１０月和四斤米") == numeric_phrases("1938年10月和四斤米")


def test_numeral_guard_allows_new_phrases_but_not_changed_input_values() -> None:
    assert numerals_preserved("四斤米", "四斤米装在一个袋子里")
    assert numerals_preserved("桌上放着一个杯子", "桌上放着个杯子")
    assert not numerals_preserved("万人大军", "上万人组成的大军")
    assert numerals_preserved("近百名来客", "将近一百人来了")
