from pathlib import Path

from scripts.prepare_corpus import prepare_file, split_text


def test_split_text_keeps_content_and_respects_maximum() -> None:
    text = "甲" * 120 + "。" + "乙" * 120 + "。" + "丙" * 120 + "。"
    chunks = split_text(text, target_chars=200, max_chars=250)

    assert "".join(chunks) == text
    assert all(len(chunk) <= 250 for chunk in chunks)


def test_prepare_file_uses_work_and_zero_padded_ids(tmp_path: Path) -> None:
    source = tmp_path / "自备作品.txt"
    source.write_text("第一句。第二句。", encoding="utf-8")

    rows = prepare_file(source)

    assert rows[0].id == "自备作品_0001"
    assert rows[0].work == "自备作品"
    assert rows[0].vernacular == ""
