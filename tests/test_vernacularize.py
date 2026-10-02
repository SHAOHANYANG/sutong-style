from pathlib import Path

from scripts.chunk_corpus import CorpusChunk
from scripts.vernacularize import (
    Pair,
    generate_pair,
    numerals_preserved,
    numeric_phrases,
    paragraph_count,
    run_batch,
    text_similarity,
)
from tests.fakes import FakeGenerator


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
