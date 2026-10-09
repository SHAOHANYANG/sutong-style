"""Check tracked files and Git history for long verbatim Chinese spans."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

LOGGER = structlog.get_logger()
MAX_FILE_BYTES = 5 * 1024 * 1024
CJK_MIN = 0x3400
CJK_MAX = 0x9FFF


@dataclass(frozen=True)
class CorpusSources:
    original: frozenset[str]
    vernacular: frozenset[str]


@dataclass(frozen=True)
class SourceStats:
    pairs_records: int
    original_windows: int
    vernacular_windows: int
    raw_files: int


@dataclass(frozen=True)
class Hit:
    location: str
    line: int | None
    count: int
    hashes: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class SkippedFile:
    location: str
    reason: str
    size_bytes: int | None


def chinese_only(text: str) -> str:
    return "".join(char for char in text if CJK_MIN <= ord(char) <= CJK_MAX)


def windows(text: str, min_chars: int) -> frozenset[str]:
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    compact = chinese_only(text)
    return frozenset(
        compact[index : index + min_chars] for index in range(max(0, len(compact) - min_chars + 1))
    )


def window_hash(fragment: str) -> str:
    return hashlib.sha256(fragment.encode("utf-8")).hexdigest()[:12]


def build_sources(
    pairs_path: Path,
    raw_dir: Path,
    min_chars: int,
    chunks_path: Path | None = None,
) -> tuple[CorpusSources, SourceStats, list[SkippedFile]]:
    original: set[str] = set()
    vernacular: set[str] = set()
    records = 0
    skipped: list[SkippedFile] = []
    if pairs_path.exists():
        with pairs_path.open(encoding="utf-8") as handle:
            for _line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                original.update(windows(str(record.get("original", "")), min_chars))
                vernacular.update(windows(str(record.get("vernacular", "")), min_chars))
                records += 1
    # Chunks that never became a pair are original prose too.
    if chunks_path is not None and chunks_path.exists():
        with chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    original.update(windows(str(json.loads(line).get("original", "")), min_chars))
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file():
                continue
            data, skip = read_text_bytes(path, str(path))
            if skip is not None:
                skipped.append(skip)
            elif data is not None:
                original.update(windows(data, min_chars))
    return (
        CorpusSources(frozenset(original), frozenset(vernacular)),
        SourceStats(
            records,
            len(original),
            len(vernacular),
            len(list(raw_dir.rglob("*.txt"))) if raw_dir.exists() else 0,
        ),
        skipped,
    )


def read_text_bytes(path: Path, location: str) -> tuple[str | None, SkippedFile | None]:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return None, SkippedFile(location, "larger_than_5MB", size)
    data = path.read_bytes()
    if b"\x00" in data:
        return None, SkippedFile(location, "binary", size)
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, SkippedFile(location, "not_utf8_text", size)


def line_for_compact_index(text: str, compact_index: int) -> int:
    seen = 0
    for line, character in enumerate(text.splitlines(keepends=True), start=1):
        for char in character:
            if CJK_MIN <= ord(char) <= CJK_MAX:
                if seen == compact_index:
                    return line
                seen += 1
    return text.count("\n") + 1


def find_hits(
    text: str, source: frozenset[str], min_chars: int, location: str, kind: str
) -> list[Hit]:
    compact = chinese_only(text)
    counts: dict[int, int] = {}
    hashes: dict[int, set[str]] = {}
    for index in range(max(0, len(compact) - min_chars + 1)):
        fragment = compact[index : index + min_chars]
        if fragment in source:
            line = line_for_compact_index(text, index)
            counts[line] = counts.get(line, 0) + 1
            hashes.setdefault(line, set()).add(window_hash(fragment))
    if not counts:
        return []
    return [
        Hit(
            location=location,
            line=line,
            count=counts[line],
            hashes=tuple(sorted(hashes[line])),
            kind=kind,
        )
        for line in sorted(counts)
    ]


def git_output(args: Sequence[str], cwd: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def tracked_paths(repo: Path) -> list[Path]:
    raw = git_output(["ls-files", "-z"], repo)
    return [repo / value.decode("utf-8") for value in raw.split(b"\x00") if value]


def history_blobs(repo: Path) -> Iterator[tuple[str, str, str]]:
    commits = git_output(["rev-list", "--all"], repo).decode("ascii").splitlines()
    seen: set[str] = set()
    for commit in commits:
        tree = git_output(["ls-tree", "-r", "-z", commit], repo)
        for entry in tree.split(b"\x00"):
            if not entry:
                continue
            header, path_bytes = entry.split(b"\t", 1)
            _mode, object_type, object_id = header.split()
            if object_type.decode("ascii") != "blob" or object_id.decode("ascii") in seen:
                continue
            blob_id = object_id.decode("ascii")
            seen.add(blob_id)
            yield commit, path_bytes.decode("utf-8"), blob_id


def blob_text(repo: Path, blob_id: str) -> tuple[str | None, str | None, int]:
    process = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input=f"{blob_id}\n".encode("ascii"),
        check=True,
        capture_output=True,
    )
    header, payload = process.stdout.split(b"\n", 1)
    fields = header.split()
    if len(fields) != 3 or fields[1] != b"blob":
        return None, "not_blob", 0
    size = int(fields[2])
    payload = payload[:size]
    if size > MAX_FILE_BYTES:
        return None, "larger_than_5MB", size
    if b"\x00" in payload:
        return None, "binary", size
    try:
        return payload.decode("utf-8"), None, size
    except UnicodeDecodeError:
        return None, "not_utf8_text", size


def scan_files(
    paths: Iterable[tuple[str, Path]], sources: CorpusSources, min_chars: int
) -> tuple[list[Hit], list[SkippedFile]]:
    hits: list[Hit] = []
    skipped: list[SkippedFile] = []
    for location, path in paths:
        text, skip = read_text_bytes(path, location)
        if skip is not None:
            skipped.append(skip)
        elif text is not None:
            hits.extend(find_hits(text, sources.original, min_chars, location, "original"))
            hits.extend(find_hits(text, sources.vernacular, min_chars, location, "vernacular"))
    return hits, skipped


def result_payload(
    *,
    sources: CorpusSources,
    source_stats: SourceStats,
    hits: list[Hit],
    skipped: list[SkippedFile],
    history_blob_count: int,
    min_chars: int,
    history: bool,
) -> dict[str, Any]:
    original_hits = [hit for hit in hits if hit.kind == "original"]
    vernacular_hits = [hit for hit in hits if hit.kind == "vernacular"]
    return {
        "min_chars": min_chars,
        "history": history,
        "source": {
            "pairs_records": source_stats.pairs_records,
            "raw_files": source_stats.raw_files,
            "original_windows": source_stats.original_windows,
            "vernacular_windows": source_stats.vernacular_windows,
        },
        "scanned": {
            "tracked_files": None,
            "history_blobs": history_blob_count,
        },
        "hits": {
            "original": [hit.__dict__ for hit in original_hits],
            "vernacular": [hit.__dict__ for hit in vernacular_hits],
        },
        "skipped": [skip.__dict__ for skip in skipped],
        "exit_code": 1 if original_hits else 2 if vernacular_hits else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("corpus/raw"))
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    sources, source_stats, skipped = build_sources(
        args.pairs, args.raw_dir, args.min_chars, args.chunks
    )
    if not sources.original:
        # Nothing to compare against must not read as a clean result.
        raise SystemExit(
            f"没有读到任何原文片段（{args.pairs}、{args.chunks}、{args.raw_dir}），无法判断是否泄漏"
        )
    paths = [(str(path.relative_to(repo)), path) for path in tracked_paths(repo)]
    hits, tracked_skipped = scan_files(paths, sources, args.min_chars)
    skipped.extend(tracked_skipped)
    blob_count = 0
    if args.history:
        for commit, path, blob_id in history_blobs(repo):
            blob_count += 1
            text, reason, size = blob_text(repo, blob_id)
            location = f"{commit}:{path}"
            if reason is not None:
                skipped.append(SkippedFile(location, reason, size))
            elif text is not None:
                hits.extend(find_hits(text, sources.original, args.min_chars, location, "original"))
                hits.extend(
                    find_hits(text, sources.vernacular, args.min_chars, location, "vernacular")
                )
    payload = result_payload(
        sources=sources,
        source_stats=source_stats,
        hits=hits,
        skipped=skipped,
        history_blob_count=blob_count,
        min_chars=args.min_chars,
        history=args.history,
    )
    payload["scanned"]["tracked_files"] = len(paths)
    if args.report is not None:
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    LOGGER.info(
        "corpus_leak_check",
        tracked_files=len(paths),
        history_blobs=blob_count,
        original_hits=len([hit for hit in hits if hit.kind == "original"]),
        vernacular_hits=len([hit for hit in hits if hit.kind == "vernacular"]),
        skipped=len(skipped),
        exit_code=payload["exit_code"],
    )
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
