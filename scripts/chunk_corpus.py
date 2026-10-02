"""Split the six user-supplied works into stable, copyright-restricted chunks."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import structlog
from pydantic import BaseModel

from scripts.prepare_corpus import write_jsonl

LOGGER = structlog.get_logger()
ANTHOLOGY_FILENAME = "妻妾成群.txt"
ANTHOLOGY_WORKS = ("妻妾成群", "妇女生活", "另一种妇女生活", "园艺")
# Longest first is intentional: 另一种妇女生活 must be tested before 妇女生活.
TITLE_PATTERN = re.compile(
    "|".join(re.escape(title) for title in sorted(ANTHOLOGY_WORKS, key=len, reverse=True))
)
SECTION_HEADING = re.compile(r"^第[零〇一二三四五六七八九十百0-9]+[章节回部]$")
SENTENCE_END = re.compile(r"(?<=[。！？；])")
CLAUSE_END = re.compile(r"(?<=[，、：])")
CJK = re.compile(r"[\u3400-\u9fff]")
TRADITIONAL_RESIDUE = str.maketrans(
    {
        "麽": "么",
        "麼": "么",
        "裏": "里",
        "裡": "里",
        "牠": "它",
    }
)


class CorpusChunk(BaseModel):
    """One source-side chunk before vernacularization."""

    id: str
    work: str
    idx: int
    original: str


def cjk_length(text: str) -> int:
    """Count Chinese characters for the 200-400 character contract."""
    return len(CJK.findall(text))


def clean_text(text: str) -> str:
    """Remove known conversion residue while preserving all quotation marks."""
    return text.replace("\r\n", "\n").replace("\r", "\n").translate(TRADITIONAL_RESIDUE)


def split_anthology(text: str) -> dict[str, str]:
    """Split the four-work anthology using title-only lines, never substring search."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in clean_text(text).splitlines():
        stripped = line.strip()
        title_match = TITLE_PATTERN.fullmatch(stripped)
        if title_match:
            current = title_match.group(0)
            if current in sections:
                raise ValueError(f"重复的作品标题行: {current}")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    missing = set(ANTHOLOGY_WORKS).difference(sections)
    if missing:
        raise ValueError(f"合集缺少标题行: {sorted(missing)}")
    return {work: "\n".join(sections[work]).strip() for work in ANTHOLOGY_WORKS}


def _split_long_atom(text: str, *, max_chars: int) -> list[str]:
    if cjk_length(text) <= max_chars:
        return [text.strip()]
    clauses = [part.strip() for part in CLAUSE_END.split(text) if part.strip()]
    if len(clauses) > 1:
        result: list[str] = []
        current = ""
        for clause in clauses:
            if current and cjk_length(current + clause) > max_chars:
                result.append(current)
                current = clause
            else:
                current += clause
        if current:
            result.append(current)
        if all(cjk_length(part) <= max_chars for part in result):
            return result

    result = []
    current_chars: list[str] = []
    current_cjk = 0
    for char in text.strip():
        if CJK.fullmatch(char):
            current_cjk += 1
        current_chars.append(char)
        if current_cjk >= max_chars:
            result.append("".join(current_chars).strip())
            current_chars = []
            current_cjk = 0
    if current_chars:
        result.append("".join(current_chars).strip())
    return result


def _paragraph_atoms(paragraph: str, *, max_chars: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_END.split(paragraph) if part.strip()]
    atoms: list[str] = []
    for sentence in sentences:
        atoms.extend(_split_long_atom(sentence, max_chars=max_chars))
    return atoms


def _sections(text: str) -> list[list[str]]:
    """Return paragraph sections; headings and ※※※ become hard preferred boundaries."""
    sections: list[list[str]] = []
    paragraphs: list[str] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraphs.append("\n".join(paragraph_lines).strip())
            paragraph_lines.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append(paragraphs.copy())
            paragraphs.clear()

    for line in clean_text(text).splitlines():
        stripped = line.strip()
        if stripped == "※※※" or SECTION_HEADING.fullmatch(stripped):
            flush_section()
        elif not stripped or stripped == "苏童":
            flush_paragraph()
        else:
            paragraph_lines.append(stripped)
    flush_section()
    return sections


def _join_atoms(atoms: list[tuple[str, bool]]) -> str:
    output = ""
    for text, starts_paragraph in atoms:
        separator = "\n" if output and starts_paragraph else ""
        output += separator + text
    return output


def _pack_section(
    paragraphs: list[str], *, target_chars: int, min_chars: int, max_chars: int
) -> list[str]:
    atoms: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        parts = _paragraph_atoms(paragraph, max_chars=max_chars)
        atoms.extend((part, index == 0) for index, part in enumerate(parts))

    chunks: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    for atom in atoms:
        candidate = _join_atoms([*current, atom])
        current_length = cjk_length(_join_atoms(current))
        candidate_length = cjk_length(candidate)
        if current and (
            candidate_length > max_chars
            or (current_length >= min_chars and candidate_length > target_chars)
        ):
            chunks.append(current)
            current = [atom]
        else:
            current.append(atom)
    if current:
        chunks.append(current)

    if len(chunks) >= 2 and cjk_length(_join_atoms(chunks[-1])) < min_chars:
        merged = [*chunks[-2], *chunks[-1]]
        if cjk_length(_join_atoms(merged)) <= max_chars:
            chunks[-2:] = [merged]

    return [_join_atoms(chunk).strip() for chunk in chunks if _join_atoms(chunk).strip()]


def _as_atoms(text: str, *, max_chars: int) -> list[tuple[str, bool]]:
    atoms: list[tuple[str, bool]] = []
    for paragraph in text.splitlines():
        parts = _paragraph_atoms(paragraph, max_chars=max_chars)
        atoms.extend((part, index == 0) for index, part in enumerate(parts))
    return atoms


def _rebalance_short_chunks(chunks: list[str], *, min_chars: int, max_chars: int) -> list[str]:
    """Move a nearby sentence boundary so every final chunk reaches the minimum."""
    balanced: list[str] = []
    for chunk in chunks:
        if cjk_length(chunk) >= min_chars or not balanced:
            balanced.append(chunk)
            continue

        previous = balanced.pop()
        atoms = _as_atoms(previous + "\n" + chunk, max_chars=max_chars)
        candidates: list[tuple[int, int]] = []
        for boundary in range(1, len(atoms)):
            left_length = cjk_length(_join_atoms(atoms[:boundary]))
            right_length = cjk_length(_join_atoms(atoms[boundary:]))
            if min_chars <= left_length <= max_chars and min_chars <= right_length <= max_chars:
                candidates.append((abs(left_length - right_length), boundary))
        if not candidates:
            combined = previous + "\n" + chunk
            target = cjk_length(combined) // 2
            seen = 0
            split_at = 0
            for index, char in enumerate(combined, start=1):
                if CJK.fullmatch(char):
                    seen += 1
                if seen >= target:
                    split_at = index
                    break
            left = combined[:split_at].strip()
            right = combined[split_at:].strip()
            right_length = cjk_length(right)
            if (
                min_chars <= cjk_length(left) <= max_chars
                and min_chars <= right_length <= max_chars
            ):
                balanced.extend((left, right))
            else:
                balanced.extend((previous, chunk))
            continue
        _, boundary = min(candidates)
        balanced.extend(
            (_join_atoms(atoms[:boundary]).strip(), _join_atoms(atoms[boundary:]).strip())
        )
    return balanced


def chunk_work(
    work: str,
    text: str,
    *,
    target_chars: int = 260,
    min_chars: int = 200,
    max_chars: int = 400,
) -> list[CorpusChunk]:
    """Chunk one work with paragraph preservation and preferred section boundaries."""
    packed = [
        chunk
        for section in _sections(text)
        for chunk in _pack_section(
            section,
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
        )
    ]
    chunks = _rebalance_short_chunks(packed, min_chars=min_chars, max_chars=max_chars)
    return [
        CorpusChunk(id=f"{work}_{idx:04d}", work=work, idx=idx, original=chunk)
        for idx, chunk in enumerate(chunks, start=1)
    ]


def load_works(input_dir: Path) -> dict[str, str]:
    """Load exactly the restored three files and expose all six works."""
    anthology_path = input_dir / ANTHOLOGY_FILENAME
    works = split_anthology(anthology_path.read_text(encoding="utf-8"))
    for work in ("我的帝王生涯", "罂粟之家"):
        path = input_dir / f"{work}.txt"
        text = clean_text(path.read_text(encoding="utf-8"))
        lines = text.splitlines()
        if lines and lines[0].strip() == work:
            lines = lines[1:]
        works[work] = "\n".join(lines).strip()
    return works


def build_chunks(input_dir: Path, *, target_chars: int = 260) -> list[CorpusChunk]:
    """Build chunks for all six works in a deterministic order."""
    works = load_works(input_dir)
    order: Iterable[str] = (*ANTHOLOGY_WORKS, "罂粟之家", "我的帝王生涯")
    return [
        chunk
        for work in order
        for chunk in chunk_work(work, works[work], target_chars=target_chars)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把六部自备作品切为 200–400 字 chunk")
    parser.add_argument("--input-dir", type=Path, default=Path("corpus/raw"))
    parser.add_argument("--output", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--target-chars", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = build_chunks(args.input_dir, target_chars=args.target_chars)
    write_jsonl(chunks, args.output)
    by_work: dict[str, int] = {}
    for chunk in chunks:
        by_work[chunk.work] = by_work.get(chunk.work, 0) + 1
    lengths = [cjk_length(chunk.original) for chunk in chunks]
    LOGGER.info(
        "corpus_chunked",
        total=len(chunks),
        by_work=by_work,
        min_chars=min(lengths),
        max_chars=max(lengths),
        output=str(args.output),
    )
    return 0 if 877 <= len(chunks) <= 937 and all(chunk.original for chunk in chunks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
