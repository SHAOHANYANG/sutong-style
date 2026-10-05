"""Entity, numeral, and title facts. Numerals reuse the vernacularizer's extractor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Literal, Protocol

import jieba.posseg as pseg
from pydantic import BaseModel, Field

from eval.metrics import entity_recall, hallucination_rate, numeral_recall
from scripts.vernacularize import PROPER_NOUN_FLAGS, extract_quantities

# Offices and ranked appellations. Synonyms are intentionally empty: 太医 is not 宫监.
TITLES: frozenset[str] = frozenset(
    {
        "太医",
        "宫监",
        "丞相",
        "钦差",
        "长工",
        "管家",
        "皇帝",
        "皇后",
        "王后",
        "贵妃",
        "太监",
        "侍卫",
        "将军",
        "大少爷",
        "大太太",
        "二太太",
        "三太太",
        "四太太",
        "二姨太",
        "三姨太",
        "四姨太",
    }
)
GAZETTEER_MIN_COUNT = 3


class EntityTagger(Protocol):
    """Injected named-entity source. Tests do not need a real segmenter."""

    def entities(self, text: str) -> set[str]: ...


class Facts(BaseModel):
    """Normalized facts for one text. Numeral keys come from extract_quantities."""

    entities: set[str] = Field(default_factory=set)
    numerals: set[str] = Field(default_factory=set)
    titles: set[str] = Field(default_factory=set)


class Violation(BaseModel):
    """One concrete fidelity miss. Title replacements name both sides."""

    kind: Literal["title"]
    expected: str
    actual: str | None = None


class FidelityReport(BaseModel):
    """Three SPEC rates plus title violations."""

    entity_recall: float
    numeral_recall: float
    hallucination_rate: float
    violations: list[Violation] = Field(default_factory=list)


class JiebaTagger:
    """Proper-noun tags already used by the corpus tools."""

    def entities(self, text: str) -> set[str]:
        return {
            word
            for word, flag in pseg.cut(text)
            if flag in PROPER_NOUN_FLAGS and len(word) >= 2 and word not in TITLES
        }


def numeral_keys(text: str) -> set[str]:
    """Cardinal, ordinal, and unparsed keys from the shared quantity extractor."""
    profile = extract_quantities(text)
    return (
        set(profile.cardinals) | {f"第{item}" for item in profile.ordinals} | set(profile.unparsed)
    )


def extract_facts(
    text: str,
    gaz: set[str],
    tagger: EntityTagger | None = None,
) -> Facts:
    """Collect gazetteer hits, tagger hits, normalized numerals, and known titles."""
    titles = {title for title in TITLES if title in text}
    tagged = (tagger or JiebaTagger()).entities(text)
    entities = {word for word in tagged if word not in titles and len(word) >= 2}
    entities.update(word for word in gaz if word in text and word not in titles and len(word) >= 2)
    return Facts(entities=entities, numerals=numeral_keys(text), titles=titles)


def title_violations(source: set[str], output: set[str]) -> list[Violation]:
    """Pair each missing title with a newly introduced one, in sorted order."""
    missing = sorted(source - output)
    extra = sorted(output - source)
    return [
        Violation(kind="title", expected=title, actual=extra[index] if index < len(extra) else None)
        for index, title in enumerate(missing)
    ]


def assess(
    source: str,
    output: str,
    gaz: set[str] | None = None,
    tagger: EntityTagger | None = None,
) -> FidelityReport:
    """Score one pair. Semantic paraphrases that keep names and numbers do not crash."""
    gazetteer = gaz or set()
    source_facts = extract_facts(source, gazetteer, tagger)
    output_facts = extract_facts(output, gazetteer, tagger)
    return FidelityReport(
        entity_recall=entity_recall(source_facts.entities, output_facts.entities),
        numeral_recall=numeral_recall(source_facts.numerals, output_facts.numerals),
        hallucination_rate=hallucination_rate(source_facts.entities, output_facts.entities),
        violations=title_violations(source_facts.titles, output_facts.titles),
    )


def build_entity_gazetteer(
    texts: Iterable[str],
    *,
    min_count: int = GAZETTEER_MIN_COUNT,
    tagger: EntityTagger | None = None,
) -> set[str]:
    """Frequent proper nouns from original texts. Titles stay on the title list."""
    counts: Counter[str] = Counter()
    reader = tagger or JiebaTagger()
    for text in texts:
        counts.update(reader.entities(text))
    return {word for word, count in counts.items() if count >= min_count and len(word) >= 2}
