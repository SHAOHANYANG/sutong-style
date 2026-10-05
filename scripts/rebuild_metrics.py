"""Pure T0.0 generation diagnostics; not the future formal T0.3 evaluator."""

from __future__ import annotations

import re
from decimal import Decimal

import cn2an
import jieba.posseg as pseg
from pydantic import BaseModel, Field
from sacrebleu.metrics.bleu import BLEU

from scripts.chunk_corpus import CJK, cjk_length
from scripts.vernacularize import GRAMMATICAL_ONE, PROPER_NOUN_FLAGS, text_similarity

QUANTITY_UNITS = re.compile(
    r"^(?:公斤|公里|大洋|年|月|日|号|岁|个|位|名|人|口|只|条|件|家|间|所|匹|头|斤|两|克|"
    r"里|亩|州|县|国|军|代|届|次|回|遍|天|夜|块|元|角|分|束|朵|颗|粒|枚|辆|艘|顶|把|柄|"
    r"张|页|本|封|桌|场|杯|碗|盘)"
)
NUMBER = re.compile(
    r"[0-9０-９零〇一二两三四五六七八九十百千万亿]+(?:[.点][0-9０-９零〇一二两三四五六七八九]+)?"
)
WIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
KNOWN_ENTITIES = (
    "颂莲",
    "陈佐千",
    "梅珊",
    "毓如",
    "飞浦",
    "沉草",
    "端白",
    "燮国",
    "枫杨树",
    "娴",
    "芝",
    "箫",
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
    "夫人",
    "姨太太",
    "大太太",
    "二太太",
    "三太太",
    "四太太",
    "二姨太",
    "三姨太",
    "四姨太",
)


class NumericFacts(BaseModel):
    values: set[str] = Field(default_factory=set)
    grammatical_waivers: list[str] = Field(default_factory=list)
    unparsed: list[str] = Field(default_factory=list)
    chinese_forms: list[str] = Field(default_factory=list)


class TextMetrics(BaseModel):
    pinc1: float | None
    pinc2: float | None
    pinc3: float | None
    pinc4: float | None
    pinc6: float | None
    pinc_mean_1_to_4: float | None
    sbleu: float
    gram6: float | None
    gram4: float | None
    similarity: float
    length_ratio: float
    entity_recall: float
    entity_denominator: int
    missing_entities: list[str]
    numeral_recall: float
    numeral_denominator: int
    missing_values: list[str]
    added_values: list[str]
    chinese_quantity_forms: list[str]
    numeral_parse_errors: list[str]
    grammatical_waivers: list[str]
    banned_residue: list[str]


def ngram_overlap(original: str, vernacular: str, n: int) -> float | None:
    """Source-side recall of distinct Chinese-only contiguous n-grams."""
    if n < 1:
        raise ValueError("n must be positive")
    source = "".join(CJK.findall(original))
    target = "".join(CJK.findall(vernacular))
    source_grams = {source[i : i + n] for i in range(len(source) - n + 1)}
    if not source_grams:
        return None
    target_grams = {target[i : i + n] for i in range(len(target) - n + 1)}
    return len(source_grams & target_grams) / len(source_grams)


def pinc(original: str, output: str, n: int) -> float | None:
    """Candidate-occurrence denominator; binary source membership, Chinese characters."""
    if n < 1:
        raise ValueError("n must be positive")
    source = "".join(CJK.findall(original))
    candidate = "".join(CJK.findall(output))
    if len(candidate) < n:
        return None
    source_grams = {source[i : i + n] for i in range(len(source) - n + 1)}
    grams = [candidate[i : i + n] for i in range(len(candidate) - n + 1)]
    return sum(gram not in source_grams for gram in grams) / len(grams)


def source_bleu(original: str, output: str) -> float:
    """Sentence sBLEU on Chinese characters, exp smoothing, effective order; 0-100."""
    scorer = BLEU(tokenize="none", smooth_method="exp", effective_order=True)
    source = " ".join(CJK.findall(original))
    candidate = " ".join(CJK.findall(output))
    return float(scorer.sentence_score(candidate, [source]).score)


def entity_candidates(original: str) -> list[str]:
    tagged = {
        word for word, flag in pseg.cut(original) if flag in PROPER_NOUN_FLAGS and len(word) >= 2
    }
    return sorted(tagged | {word for word in KNOWN_ENTITIES if word in original})


def mask_entities(text: str, entities: list[str]) -> str:
    for entity in sorted(entities, key=lambda value: (-len(value), value)):
        text = text.replace(entity, " " * len(entity))
    return text


def numeric_values(text: str, entities: list[str] | None = None) -> NumericFacts:
    """Normalize values, not typography; ambiguous single characters are not quantities."""
    cleaned = mask_entities(text.translate(WIDTH), entities or [])
    # Idiomatic 一/二/万 are lexical style, not quantitative facts.
    for word, flag in pseg.cut(cleaned):
        if flag == "i":
            cleaned = cleaned.replace(word, " " * len(word))
    facts = NumericFacts()
    for match in NUMBER.finditer(cleaned):
        raw = match.group()
        unit = QUANTITY_UNITS.match(cleaned[match.end() :])
        prefix = cleaned[max(0, match.start() - 1) : match.start()]
        is_arabic = bool(re.search(r"[0-9]", raw))
        if not is_arabic and len(raw) == 1 and not unit and prefix not in {"第", "初"}:
            continue
        phrase = raw + (unit.group() if unit else "")
        if GRAMMATICAL_ONE.fullmatch(phrase):
            facts.grammatical_waivers.append(phrase)
            continue
        token = "一" + raw if raw in {"十", "百", "千", "万", "亿"} else raw
        try:
            value = cn2an.cn2an(token, "smart") if not raw.isascii() else Decimal(raw)
            normalized = format(Decimal(str(value)).normalize(), "f")
        except (ValueError, TypeError):
            facts.unparsed.append(phrase)
            continue
        facts.values.add(normalized)
        if not is_arabic or re.search(r"[零〇一二两三四五六七八九十百千万亿]", raw):
            facts.chinese_forms.append(phrase)
    return facts


def measure(original: str, output: str, banned: list[str], entities: list[str]) -> TextMetrics:
    expected = numeric_values(original, entities)
    actual = numeric_values(output, entities)
    missing_entities = [word for word in entities if word not in output]
    unprotected_target = mask_entities(output, entities)
    residue = sorted(word for word in set(banned) if word in unprotected_target)
    novelty = {n: pinc(original, output, n) for n in (1, 2, 3, 4, 6)}
    standard_orders = [novelty[n] for n in (1, 2, 3, 4)]
    return TextMetrics(
        pinc1=novelty[1],
        pinc2=novelty[2],
        pinc3=novelty[3],
        pinc4=novelty[4],
        pinc6=novelty[6],
        pinc_mean_1_to_4=sum(value for value in standard_orders if value is not None) / 4
        if all(value is not None for value in standard_orders)
        else None,
        sbleu=source_bleu(original, output),
        gram6=ngram_overlap(original, output, 6),
        gram4=ngram_overlap(original, output, 4),
        similarity=text_similarity(original, output),
        length_ratio=cjk_length(output) / max(cjk_length(original), 1),
        entity_recall=1 - len(missing_entities) / len(entities) if entities else 1.0,
        entity_denominator=len(entities),
        missing_entities=missing_entities,
        numeral_recall=len(expected.values & actual.values) / len(expected.values)
        if expected.values
        else 1.0,
        numeral_denominator=len(expected.values),
        missing_values=sorted(expected.values - actual.values),
        added_values=sorted(actual.values - expected.values),
        chinese_quantity_forms=actual.chinese_forms,
        numeral_parse_errors=expected.unparsed + actual.unparsed,
        grammatical_waivers=expected.grammatical_waivers + actual.grammatical_waivers,
        banned_residue=residue,
    )


def final_failures(first: TextMetrics, final: TextMetrics) -> list[str]:
    failures: list[str] = []
    for name, minimum in (("pinc6", 0.85), ("pinc4", 0.75)):
        before, after = getattr(first, name), getattr(final, name)
        if before is None or after is None:
            failures.append(f"{name}_unassessable")
        else:
            if after < minimum - 1e-12:
                failures.append(f"{name}_below_limit")
            if before - after > 1e-12:
                failures.append(f"{name}_repair_decrease")
    if not 0.6 <= final.length_ratio <= 1.6:
        failures.append("length_out_of_range")
    if final.entity_recall < 1:
        failures.append("entity_missing")
    if final.numeral_recall < 1:
        failures.append("numeral_missing")
    if final.added_values:
        failures.append("numeral_added")
    if final.chinese_quantity_forms:
        failures.append("numeric_form_not_colloquial")
    if final.numeral_parse_errors:
        failures.append("numeral_parse_error")
    if final.banned_residue:
        failures.append("literary_residue")
    return failures
