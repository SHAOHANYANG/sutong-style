"""Create a copyright-restricted chunk skeleton from user-supplied text files.

For the restored six-work corpus, ``scripts/chunk_corpus.py`` is the canonical
entry point because it understands the four-work anthology and section markers.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import structlog
from pydantic import BaseModel

LOGGER = structlog.get_logger()
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；])")


class PairSkeleton(BaseModel):
    """A not-yet-vernacularized training pair."""

    id: str
    work: str
    idx: int
    vernacular: str = ""
    original: str
    split: str = "train"


def split_text(text: str, *, target_chars: int = 250, max_chars: int = 400) -> list[str]:
    """Split text near sentence boundaries without performing corpus-specific cleanup."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
        if len(current) >= target_chars:
            chunks.append(current)
            current = ""
    if current:
        if chunks and len(current) < 100 and len(chunks[-1]) + len(current) <= max_chars:
            chunks[-1] += current
        else:
            chunks.append(current)
    return chunks


def prepare_file(path: Path) -> list[PairSkeleton]:
    """Prepare pair skeletons for one work named after its file stem."""
    text = path.read_text(encoding="utf-8")
    return [
        PairSkeleton(id=f"{path.stem}_{idx:04d}", work=path.stem, idx=idx, original=chunk)
        for idx, chunk in enumerate(split_text(text), start=1)
    ]


def write_jsonl(rows: Iterable[BaseModel], output: Path) -> None:
    """Write pair skeletons as UTF-8 JSON Lines."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从自备文本生成 pairs.jsonl 骨架；仓库不会附带受版权保护的语料。"
    )
    parser.add_argument("input_dir", type=Path, help="包含 UTF-8 .txt 文件的目录")
    parser.add_argument(
        "--output", type=Path, default=Path("corpus/pairs.jsonl"), help="输出 JSONL 路径"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.txt"))
    if not paths:
        LOGGER.error("no_input_files", input_dir=str(args.input_dir))
        return 1
    rows = [row for path in paths for row in prepare_file(path)]
    write_jsonl(rows, args.output)
    LOGGER.info("corpus_skeleton_written", output=str(args.output), chunks=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
