"""Entity, numeral, and title facts. Numerals reuse the vernacularizer's extractor."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

import jieba
import jieba.posseg as pseg
from pydantic import BaseModel, Field

from eval.metrics import entity_recall, hallucination_rate, numeral_recall
from scripts.vernacularize import PROPER_NOUN_FLAGS, extract_quantities, numeral_key_surfaces

MANUAL_ENTITY_PATH = Path(__file__).resolve().parents[1] / "corpus" / "manual_entities.json"
# jieba tags 登基 as nrt and 入宫 as ns, so a verb flag never fires. These are the
# same class of verbal false positives already sitting in the automatic gazetteer.
VERB_FALSE_POSITIVES: frozenset[str] = frozenset(
    {
        "登基",
        "入宫",
        "登门",
        "上楼",
        "上门",
        "上路",
        "上山",
        "回京",
        "回城",
        "南伐",
        "西巡",
        "塞进",
        "涂抹",
        "张开",
        "张望",
        "张大",
        "张直",
        "张狂",
        "雨淋",
        "长大",
        "胡说",
        "呼唤",
        "测字",
        "陈述",
        "仰天长叹",
        "别以为",
        "任凭",
    }
)

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
# jieba dict.txt frequency, snapshotted before the manual user dictionary.
# 200 sits below 白痴 (242) and above the checked name 老王 (190).
COMMON_WORD_MIN_FREQ = 200
_USERDICT_READY = False
_MANUAL_ENTITIES: list[ManualEntity] | None = None
_GENERAL_FREQ: dict[str, int] | None = None
_STABLE_TOKEN: dict[str, bool] = {}


class ManualEntity(BaseModel):
    """One hand-checked name. The string is also the jieba user-dictionary token."""

    name: str
    kind: Literal["person", "place"]
    works: list[str]


def load_manual_entities() -> list[ManualEntity]:
    """Read the committed name list. No sentences are stored in that file."""
    global _MANUAL_ENTITIES
    if _MANUAL_ENTITIES is None:
        payload = json.loads(MANUAL_ENTITY_PATH.read_text(encoding="utf-8"))
        _MANUAL_ENTITIES = [ManualEntity.model_validate(item) for item in payload["entities"]]
    return _MANUAL_ENTITIES


def general_frequencies() -> dict[str, int]:
    """jieba's general-vocabulary counts, before any story names are inserted."""
    global _GENERAL_FREQ
    if _GENERAL_FREQ is None:
        jieba.initialize()
        _GENERAL_FREQ = dict(jieba.dt.FREQ)
    return _GENERAL_FREQ


def ensure_manual_userdict() -> None:
    """Load the manual names into jieba once, so compounds stay one token."""
    global _USERDICT_READY
    if _USERDICT_READY:
        return
    general_frequencies()
    lines = [
        f"{item.name} 100000 {'nr' if item.kind == 'person' else 'ns'}"
        for item in load_manual_entities()
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".dict",
        delete=False,
    ) as handle:
        handle.write("\n".join(lines) + "\n")
        dict_path = handle.name
    jieba.load_userdict(dict_path)
    _USERDICT_READY = True


class EntityTagger(Protocol):
    """Injected named-entity source. Tests do not need a real segmenter."""

    def entities(self, text: str) -> set[str]: ...


class Facts(BaseModel):
    """Normalized facts for one text. Numeral keys come from extract_quantities."""

    entities: set[str] = Field(default_factory=set)
    numerals: set[str] = Field(default_factory=set)
    titles: set[str] = Field(default_factory=set)


ViolationKind = Literal[
    "entity_missing",
    "numeral_missing",
    "entity_hallucination",
    "title",
]


class Violation(BaseModel):
    """One concrete fidelity miss. Surfaces are spans from the source texts."""

    kind: ViolationKind
    expected: str | None = None
    actual: str | None = None


class FidelityReport(BaseModel):
    """Three SPEC rates plus the full violation list derived from the same facts."""

    entity_recall: float
    numeral_recall: float
    hallucination_rate: float
    violations: list[Violation] = Field(default_factory=list)


class JiebaTagger:
    """Proper-noun tags. The manual user dictionary is loaded before cutting."""

    def entities(self, text: str) -> set[str]:
        ensure_manual_userdict()
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


def manual_mentions(text: str) -> set[str]:
    """Longest-match the hand-checked names, including single-character protagonists."""
    ensure_manual_userdict()
    names = sorted((item.name for item in load_manual_entities()), key=len, reverse=True)
    occupied = bytearray(len(text))
    found: set[str] = set()
    for name in names:
        start = 0
        while True:
            index = text.find(name, start)
            if index < 0:
                break
            end = index + len(name)
            if not any(occupied[index:end]):
                found.add(name)
                occupied[index:end] = b"\x01" * len(name)
            start = index + 1
    return found


def _stable_token(word: str) -> bool:
    """True when jieba keeps this string as one word, not a context fragment."""
    cached = _STABLE_TOKEN.get(word)
    if cached is None:
        ensure_manual_userdict()
        cached = list(jieba.cut(word)) == [word]
        _STABLE_TOKEN[word] = cached
    return cached


def supplementary_words(words: Iterable[str]) -> set[str]:
    """Gazetteer entries the supplementary channel can still emit."""
    return {word for word in words if not _supplement_rejected(word, set())}


def _supplement_rejected(word: str, manual_hits: set[str]) -> bool:
    """Drop verbs, common nouns, unstable fragments, and pieces of a longer name."""
    if word in VERB_FALSE_POSITIVES or word in TITLES or len(word) < 2:
        return True
    if general_frequencies().get(word, 0) >= COMMON_WORD_MIN_FREQ:
        return True
    if not _stable_token(word):
        return True
    if any(word != name and word in name for name in manual_hits):
        return True
    manual_names = {item.name for item in load_manual_entities()}
    return any(word != name and name in word and word not in manual_names for name in manual_names)


def extract_facts(
    text: str,
    gaz: set[str],
    tagger: EntityTagger | None = None,
) -> Facts:
    """Manual names are primary. Counted, stable gazetteer words only supplement."""
    titles = {title for title in TITLES if title in text}
    manual = manual_mentions(text)
    # A live jieba tag joins only when the train gazetteer already counted it.
    # One-off fragments such as 宫面圣 never reach GAZETTEER_MIN_COUNT.
    counted = {word for word in (tagger or JiebaTagger()).entities(text) if word in gaz}
    supplement = {
        word
        for word in set(gaz) | counted
        if word in text and word in gaz and not _supplement_rejected(word, manual)
    }
    return Facts(
        entities=(manual | supplement) - titles,
        numerals=numeral_keys(text),
        titles=titles,
    )


def title_violations(source: set[str], output: set[str]) -> list[Violation]:
    """Pair each missing title with a newly introduced one, in sorted order."""
    missing = sorted(source - output)
    extra = sorted(output - source)
    return [
        Violation(kind="title", expected=title, actual=extra[index] if index < len(extra) else None)
        for index, title in enumerate(missing)
    ]


def fidelity_violations(
    source: str,
    output: str,
    source_facts: Facts,
    output_facts: Facts,
) -> list[Violation]:
    """Export concrete misses from the same fact sets that feed the three rates.

    Empty violations means entity_recall = 1, numeral_recall = 1, hallucination_rate = 0,
    and no title replacements. Surfaces are substrings of the corresponding text.
    """
    violations: list[Violation] = []
    for name in sorted(source_facts.entities - output_facts.entities):
        violations.append(Violation(kind="entity_missing", expected=name, actual=None))
    surfaces = numeral_key_surfaces(source)
    for key in sorted(source_facts.numerals - output_facts.numerals):
        expected = surfaces.get(key)
        if expected is None:
            raise RuntimeError(f"数值键 {key!r} 在输入中没有对应表面形式")
        violations.append(Violation(kind="numeral_missing", expected=expected, actual=None))
    for name in sorted(output_facts.entities - source_facts.entities):
        violations.append(Violation(kind="entity_hallucination", expected=None, actual=name))
    violations.extend(title_violations(source_facts.titles, output_facts.titles))
    return violations


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
        violations=fidelity_violations(source, output, source_facts, output_facts),
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
