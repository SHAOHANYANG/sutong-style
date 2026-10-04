"""Deterministic raw-corpus short-term extraction, without source sentence export."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import jieba
import jieba.posseg as pseg
import structlog
from pydantic import BaseModel, Field

from scripts.chunk_corpus import clean_text

LITERARY_CANDIDATES = {
    "仿佛",
    "犹如",
    "宛如",
    "似乎",
    "蓦然",
    "旋即",
    "倏忽",
    "遂",
    "颇",
    "亦",
    "竟然",
    "凝视",
    "伫立",
    "踌躇",
    "蹒跚",
    "踯躅",
    "凄凉",
    "凄然",
    "怅然",
    "愕然",
    "悻悻",
    "苍茫",
    "寂寥",
    "萧瑟",
    "缄默",
    "颓然",
    "阴郁",
    "愁怅",
    "悲戚",
    "怆然",
    "掬",
    "颤栗",
    "憔悴",
    "阑珊",
    "逡巡",
    "觊觎",
    "惶惑",
    "喟叹",
    "泯灭",
    "殆尽",
    "蓊郁",
}
COLOR_MARKERS = "绛褚赭黛靛绯褐"
COLOR_ENDINGS = ("红", "紫", "蓝", "绿", "黄", "黑", "白", "色")
TEXTURE_MARKERS = {"斑驳", "黏腻", "粘腻", "温润", "粗粝", "皲裂", "干涩", "光洁", "滑腻"}
COLLOQUIAL_FOUR = {
    "怎么回事",
    "奇怪的是",
    "想干什么",
    "发现自己",
    "闭上眼睛",
    "一切都是",
    "少年时代",
    "这么回事",
    "不感兴趣",
    "在我看来",
    "普遍认为",
}


class ReplacementTerm(BaseModel):
    term: str = Field(pattern=r"^[\u3400-\u9fff]{2,4}$")
    count: int
    categories: list[str]


class ReplacementLexicon(BaseModel):
    extraction_version: str = "raw-jieba-v1"
    jieba_version: str
    minimum_count: int = 2
    per_category_limit: int = 60
    source_sha256: dict[str, str]
    terms: list[ReplacementTerm]
    limitation: str = "Heuristic short-word candidates, not exhaustive stylistic annotation."


def classify_term(word: str, flag: str) -> set[str]:
    if not 2 <= len(word) <= 4 or flag.startswith("nr") or flag in {"ns", "nt", "nz"}:
        return set()
    categories: set[str] = set()
    if flag == "i" or (flag == "l" and len(word) == 4 and word not in COLLOQUIAL_FOUR):
        categories.add("idiom")
        if len(word) == 4:
            categories.add("four_character_expression")
    if word in LITERARY_CANDIDATES:
        categories.add("literary_word")
    if (
        any(char in word for char in COLOR_MARKERS) and word.endswith(COLOR_ENDINGS)
    ) or word in TEXTURE_MARKERS:
        categories.add("color_texture")
    return categories


def extract_lexicon(paths: list[Path]) -> ReplacementLexicon:
    counts: Counter[str] = Counter()
    categories: dict[str, set[str]] = {}
    for path in sorted(paths):
        for word, flag in pseg.cut(clean_text(path.read_text(encoding="utf-8"))):
            labels = classify_term(word, flag)
            if labels:
                counts[word] += 1
                categories.setdefault(word, set()).update(labels)
    selected: set[str] = set()
    for label in sorted({label for labels in categories.values() for label in labels}):
        candidates = sorted(
            (word for word in counts if counts[word] >= 2 and label in categories[word]),
            key=lambda word: (-counts[word], word),
        )
        selected.update(candidates[:60])
    return ReplacementLexicon(
        jieba_version=jieba.__version__,
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)
        },
        terms=[
            ReplacementTerm(term=word, count=counts[word], categories=sorted(categories[word]))
            for word in sorted(selected, key=lambda word: (-counts[word], word))
        ],
    )


def main() -> None:
    # ASCII JSON transport is safe under both UTF-8 and legacy Windows consoles.
    # Consumers decode escapes and save the resulting asset with encoding="utf-8".
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=True)])
    lexicon = extract_lexicon(list(Path("corpus/raw").glob("*.txt")))
    structlog.get_logger().info("replacement_lexicon", lexicon=lexicon.model_dump())


if __name__ == "__main__":
    main()
