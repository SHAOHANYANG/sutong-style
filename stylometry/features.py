"""20-dimensional style features. No file or network access."""

from __future__ import annotations

import re

import jieba
import numpy as np
from numpy.typing import NDArray

from stylometry.lexicon import LiteraryLexicon

FEATURE_NAMES: tuple[str, ...] = (
    "sent_len_mean",
    "sent_len_std",
    "sent_len_p90",
    "comma_ratio",
    "period_ratio",
    "quote_density",
    "dialogue_verb_density",
    "simile_density",
    "fw_的",
    "fw_了",
    "fw_着",
    "fw_地",
    "fw_而",
    "fw_其",
    "fw_之",
    "ttr",
    "avg_word_len",
    "literary_hit",
    "cjk_numeral_ratio",
    "para_density",
)
SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")
PUNCTUATION = frozenset("，。！？；：、…—～·「」『』“”‘’（）《》〈〉【】[](){}.,!?;:\"'`~")
COMMAS = frozenset("，,")
PERIODS = frozenset("。")
QUOTES = frozenset("「」『』“”‘’\"'")
DIALOGUE_VERBS = frozenset("说道问答喊嚷")
FUNCTION_WORDS = ("的", "了", "着", "地", "而", "其", "之")
SIMILES = ("好像", "仿佛", "犹如", "好比", "像", "似", "般")
CJK_NUMERALS = frozenset("零〇一二两三四五六七八九十百千万亿")
ARABIC_NUMERALS = frozenset("0123456789０１２３４５６７８９")


def _tokens(text: str) -> list[str]:
    return [token for token in jieba.lcut(text) if token.strip()]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(text) if len(part.strip()) >= 2]


def _per_hundred(count: int, n_chars: int) -> float:
    if n_chars == 0:
        return 0.0
    return count * 100.0 / n_chars


def _count_similes(text: str) -> int:
    count = 0
    index = 0
    while index < len(text):
        marker = next((item for item in SIMILES if text.startswith(item, index)), "")
        if marker:
            count += 1
            index += len(marker)
        else:
            index += 1
    return count


def extract(text: str, lexicon: LiteraryLexicon) -> NDArray[np.float64]:
    """Return a float64 vector of length 20. Empty text is all zeros."""
    if not text:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    sentences = _sentences(text)
    lengths = np.array([len(sentence) for sentence in sentences], dtype=np.float64)
    if lengths.size:
        sent_len_mean = float(lengths.mean())
        sent_len_std = float(lengths.std(ddof=0))
        sent_len_p90 = float(np.percentile(lengths, 90))
    else:
        sent_len_mean = 0.0
        sent_len_std = 0.0
        sent_len_p90 = 0.0
    punctuation = [char for char in text if char in PUNCTUATION]
    n_punct = len(punctuation)
    comma_ratio = sum(char in COMMAS for char in punctuation) / n_punct if n_punct else 0.0
    period_ratio = sum(char in PERIODS for char in punctuation) / n_punct if n_punct else 0.0
    n_chars = len(text)
    tokens = _tokens(text)
    cjk_numerals = sum(char in CJK_NUMERALS for char in text)
    arabic_numerals = sum(char in ARABIC_NUMERALS for char in text)
    numeral_total = cjk_numerals + arabic_numerals
    paragraphs = [part for part in text.split("\n") if part.strip()]
    values = {
        "sent_len_mean": sent_len_mean,
        "sent_len_std": sent_len_std,
        "sent_len_p90": sent_len_p90,
        "comma_ratio": comma_ratio,
        "period_ratio": period_ratio,
        "quote_density": _per_hundred(sum(char in QUOTES for char in text), n_chars),
        "dialogue_verb_density": _per_hundred(
            sum(char in DIALOGUE_VERBS for char in text), n_chars
        ),
        "simile_density": _per_hundred(_count_similes(text), n_chars),
        "ttr": (len(set(tokens)) / len(tokens)) if tokens else 0.0,
        "avg_word_len": (sum(len(token) for token in tokens) / len(tokens)) if tokens else 0.0,
        "literary_hit": lexicon.literary_hit_rate(tokens),
        "cjk_numeral_ratio": (cjk_numerals / numeral_total) if numeral_total else 0.0,
        "para_density": _per_hundred(len(paragraphs), n_chars),
    }
    for word in FUNCTION_WORDS:
        values[f"fw_{word}"] = _per_hundred(text.count(word), n_chars)
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)
