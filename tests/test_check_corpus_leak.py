from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.check_corpus_leak as leak
from scripts.check_corpus_leak import (
    build_sources,
    find_hits,
    main,
    read_text_bytes,
    scan_files,
    windows,
)


def write_pairs(path: Path, original: str, vernacular: str) -> None:
    path.write_text(
        json.dumps({"original": original, "vernacular": vernacular}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_windows_min_boundary_and_punctuation() -> None:
    assert windows("甲乙丙丁", 4) == {"甲乙丙丁"}
    assert windows("甲乙丙", 4) == set()
    assert windows("甲乙，\n丙丁", 4) == {"甲乙丙丁"}


def test_original_and_vernacular_hits_are_separate(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    write_pairs(pairs, "原文甲乙丙丁戊己", "白话一二三四五六")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("原文甲乙丙丁戊己", encoding="utf-8")
    sources = build_sources(pairs, tmp_path / "none", 6)[0]
    hits, skipped = scan_files([("tracked.txt", tracked)], sources, 6)
    assert not skipped
    assert [(hit.kind, hit.count) for hit in hits] == [("original", 3)]
    assert hits[0].hashes and "原文甲乙丙丁戊己" not in json.dumps(
        hits[0].__dict__, ensure_ascii=False
    )


def test_binary_is_reported_as_skipped(tmp_path: Path) -> None:
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\x00\x01")
    text, skipped = read_text_bytes(path, "binary.bin")
    assert text is None
    assert skipped is not None and skipped.reason == "binary"


def test_cli_exit_codes_and_report_does_not_contain_plaintext(tmp_path: Path, monkeypatch) -> None:
    pairs = tmp_path / "pairs.jsonl"
    write_pairs(pairs, "原创句子甲乙丙丁戊己", "白话句子一二三四五六")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("原创句子甲乙丙丁戊己", encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(leak, "tracked_paths", lambda _: [tracked])
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", "--pairs", str(pairs), "--min-chars", "6", "--report", str(report)],
    )
    assert main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["hits"]["original"]
    assert "原创句子" not in report.read_text(encoding="utf-8")
    tracked.write_text("白话句子一二三四五六", encoding="utf-8")
    assert main() == 2
    tracked.write_text("完全不同的测试文件", encoding="utf-8")
    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"\x00\x01")
    monkeypatch.setattr(leak, "tracked_paths", lambda _: [tracked, binary])
    assert main() == 0
    assert any(
        item["reason"] == "binary"
        for item in json.loads(report.read_text(encoding="utf-8"))["skipped"]
    )


def test_history_finds_deleted_blob_and_all_exit_codes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "leak.txt").write_text("历史泄漏句子甲乙丙丁戊己", encoding="utf-8")
    subprocess.run(["git", "add", "leak.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "add",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "leak.txt").unlink()
    subprocess.run(["git", "add", "-u"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "delete",
        ],
        cwd=repo,
        check=True,
    )
    pairs = tmp_path / "pairs.jsonl"
    write_pairs(pairs, "历史泄漏句子甲乙丙丁戊己", "安全白话一二三四五六")
    sources = build_sources(pairs, tmp_path / "none", 6)[0]
    from scripts.check_corpus_leak import blob_text, history_blobs

    history_hits = []
    blobs = list(history_blobs(repo))
    for commit, path, blob_id in blobs:
        text, _reason, _ = blob_text(repo, blob_id)
        if text is not None:
            history_hits.extend(
                find_hits(text, sources.original, 6, f"{commit}:{path}", "original")
            )
    assert history_hits and history_hits[0].location.endswith(":leak.txt")


def test_missing_corpus_is_an_error_not_a_clean_result(tmp_path: Path, monkeypatch) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("原创句子甲乙丙丁戊己", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(leak, "tracked_paths", lambda _: [tracked])
    monkeypatch.setattr(sys, "argv", ["check", "--pairs", str(tmp_path / "absent.jsonl")])
    with pytest.raises(SystemExit, match="没有读到任何原文片段"):
        main()


def test_chunks_without_a_pair_still_count_as_original(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    write_pairs(pairs, "配对里的原创句子甲乙", "白话句子一二三四五六")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps({"original": "没有配对的原创片段丙丁"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("没有配对的原创片段丙丁", encoding="utf-8")
    without = build_sources(pairs, tmp_path / "none", 6)[0]
    assert scan_files([("tracked.txt", tracked)], without, 6)[0] == []
    with_chunks = build_sources(pairs, tmp_path / "none", 6, chunks)[0]
    hits = scan_files([("tracked.txt", tracked)], with_chunks, 6)[0]
    assert [hit.kind for hit in hits] == ["original"]
