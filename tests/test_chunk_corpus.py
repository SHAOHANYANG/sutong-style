import hashlib
from pathlib import Path

import pytest

from scripts.chunk_corpus import (
    ANTHOLOGY_WORKS,
    SENTENCE_END,
    chunk_work,
    clean_text,
    split_anthology,
)

RAW_ANTHOLOGY = Path(__file__).resolve().parents[1] / "corpus" / "raw" / "妻妾成群.txt"
BOUNDARY_HASHES = {
    "妻妾成群": (
        "9de30dae1746bbb095252c65dc1b0b8c6b2fd81301816dd7646ba715d1264279",
        "997b1fe98b40f53420057c97feaa0d2a3add3cbcf51d131f50b9133d3182175a",
    ),
    "妇女生活": (
        "e3f8abf06120ae56fe13c869c5cace2da5e3a0514ec400f895df068001615bab",
        "efa55f1c403cd9eeaec137791dae85dc9740f80fb836daf591625232b3a0b556",
    ),
    "另一种妇女生活": (
        "6b6c031c59bed9c9886edfca588ce98b6dabaca76774dc73547547ec845b1126",
        "490fff9b4870ddc9aa5ef6068c94fb65c64f2354ad4eb4655c87373ff168b281",
    ),
    "园艺": (
        "f2150c945f2c56a0b6fb34aec8b4b8a724fbb4c936776c7e70a9e5fcf905fdda",
        "b0c4882abf3f59037ac8e1604d8f1e1a11c5550d70a841483429f4d11b959731",
    ),
}


def test_anthology_uses_title_only_lines_and_keeps_each_work_boundaries() -> None:
    sections = []
    for index, work in enumerate(ANTHOLOGY_WORKS, start=1):
        body = f"{work}首句{index}。这里把园艺当普通词，不是标题。\n\n{work}末句{index}。"
        sections.append(f"{work}\n\n{body}")
    anthology = "\n\n".join(sections)

    works = split_anthology(anthology)

    for index, work in enumerate(ANTHOLOGY_WORKS, start=1):
        assert works[work].startswith(f"{work}首句{index}。")
        assert works[work].endswith(f"{work}末句{index}。")


def test_cleaning_replaces_residue_but_preserves_quotes() -> None:
    text = "「什麽在裏面？」它问，牠没有回答。"

    assert clean_text(text) == "「什么在里面？」它问，它没有回答。"


def test_section_marker_is_a_chunk_boundary() -> None:
    left = "甲" * 210 + "。"
    right = "乙" * 210 + "。"

    chunks = chunk_work("测试", f"{left}\n\n※※※\n\n{right}")

    assert len(chunks) == 2
    assert chunks[0].original == left
    assert chunks[1].original == right


def test_restored_anthology_first_and_last_sentences_map_to_correct_work() -> None:
    if not RAW_ANTHOLOGY.exists():
        pytest.skip("copyrighted raw corpus is intentionally absent")
    works = split_anthology(RAW_ANTHOLOGY.read_text(encoding="utf-8"))

    for work, (expected_first, expected_last) in BOUNDARY_HASHES.items():
        sentences = [part.strip() for part in SENTENCE_END.split(works[work]) if part.strip()]
        actual = tuple(
            hashlib.sha256(sentence.encode("utf-8")).hexdigest()
            for sentence in (sentences[0], sentences[-1])
        )
        assert actual == (expected_first, expected_last)
