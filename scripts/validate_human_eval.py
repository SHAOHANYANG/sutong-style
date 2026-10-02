"""Validate the manually authored out-of-domain evaluation set."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import jieba.posseg as pseg
import structlog
from pydantic import BaseModel, ValidationError

LOGGER = structlog.get_logger()
CJK_RE = re.compile(r"[\u3400-\u9fff]")
DIGIT_RE = re.compile(r"[0-9０-９零〇一二两三四五六七八九十百千万亿]")
ENTITY_FLAGS = frozenset({"nr", "ns", "nt", "nz"})
EXPECTED_ROWS = 30
EXPECTED_DOMAINS = 6
ROWS_PER_DOMAIN = EXPECTED_ROWS // EXPECTED_DOMAINS


class HumanEvalRow(BaseModel):
    """One manually authored human-evaluation prompt."""

    id: str
    domain: str
    seed: str
    vernacular: str


class ValidationSummary(BaseModel):
    """Machine-readable validation outcome."""

    total: int
    pending: int
    errors: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def cjk_length(text: str) -> int:
    """Count CJK characters, excluding punctuation and whitespace."""
    return len(CJK_RE.findall(text))


def entity_count(text: str) -> int:
    """Estimate named-entity count with jieba's offline POS tagger."""
    return len({word for word, flag in pseg.cut(text) if flag in ENTITY_FLAGS and len(word) >= 2})


def validate_rows(rows: list[HumanEvalRow]) -> ValidationSummary:
    """Validate completeness, length, facts, and balanced domain distribution."""
    errors: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"应有 {EXPECTED_ROWS} 条，实际 {len(rows)} 条")

    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("id 必须唯一")

    domains = Counter(row.domain for row in rows)
    if len(domains) != EXPECTED_DOMAINS or any(
        count != ROWS_PER_DOMAIN for count in domains.values()
    ):
        distribution = dict(domains)
        errors.append(
            f"应有 {EXPECTED_DOMAINS} 个 domain，"
            f"且每个恰好 {ROWS_PER_DOMAIN} 条；实际 {distribution}"
        )

    pending = 0
    for row in rows:
        if not row.vernacular.strip():
            pending += 1
            continue
        length = cjk_length(row.vernacular)
        if not 100 <= length <= 300:
            errors.append(f"{row.id}: 中文字数应为 100–300，实际 {length}")
        entities = entity_count(row.vernacular)
        if entities < 2:
            errors.append(f"{row.id}: 实体至少 2 个，实际 {entities}")
        if not DIGIT_RE.search(row.vernacular):
            errors.append(f"{row.id}: 至少需要 1 个数字")

    return ValidationSummary(total=len(rows), pending=pending, errors=errors)


def load_rows(path: Path) -> list[HumanEvalRow]:
    """Load UTF-8 JSON Lines and report line numbers on schema failures."""
    rows: list[HumanEvalRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(HumanEvalRow.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise ValueError(f"第 {line_number} 行格式错误: {exc}") from exc
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验人工白话评测集")
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = validate_rows(load_rows(args.path))
    except (OSError, ValueError) as exc:
        LOGGER.error("human_eval_unreadable", error=str(exc))
        return 1

    if summary.errors:
        LOGGER.error("human_eval_invalid", **summary.model_dump())
        return 1
    if summary.pending:
        LOGGER.info(
            "human_eval_pending",
            message=f"{summary.pending} 条待填写",
            **summary.model_dump(),
        )
        return 0
    LOGGER.info("human_eval_valid", total=summary.total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
