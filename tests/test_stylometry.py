import json
import math
from pathlib import Path

import numpy as np
import pytest

from stylometry.distance import STD_FLOOR, StyleReference
from stylometry.features import FEATURE_NAMES, extract
from stylometry.lexicon import LiteraryLexicon, build_lexicon, lexicon_payload


def _vector(text: str, lexicon: LiteraryLexicon | None = None) -> dict[str, float]:
    values = extract(text, lexicon or LiteraryLexicon())
    return dict(zip(FEATURE_NAMES, values.tolist(), strict=True))


def test_feature_names_are_the_twenty_specified_dimensions() -> None:
    assert len(FEATURE_NAMES) == 20
    assert FEATURE_NAMES[18] == "cjk_numeral_ratio"
    assert FEATURE_NAMES[8:15] == ("fw_的", "fw_了", "fw_着", "fw_地", "fw_而", "fw_其", "fw_之")


def test_each_dimension_on_handwritten_text() -> None:
    sentence = _vector("甲乙。丙丁戊。")
    assert sentence["sent_len_mean"] == pytest.approx(2.5)
    assert sentence["sent_len_std"] == pytest.approx(0.5)
    assert sentence["sent_len_p90"] == pytest.approx(2.9)
    assert sentence["comma_ratio"] == 0.0
    assert sentence["period_ratio"] == 1.0
    assert sentence["ttr"] == pytest.approx(0.8)
    assert sentence["avg_word_len"] == pytest.approx(1.4)
    assert sentence["para_density"] == pytest.approx(100 / 7)

    ratios = _vector("甲，乙。")
    assert ratios["comma_ratio"] == pytest.approx(0.5)
    assert ratios["period_ratio"] == pytest.approx(0.5)
    assert ratios["sent_len_mean"] == pytest.approx(3.0)
    assert ratios["sent_len_std"] == 0.0

    quotes = _vector("「甲乙」")
    assert quotes["quote_density"] == pytest.approx(50.0)

    dialogue = _vector("他说道。")
    assert dialogue["dialogue_verb_density"] == pytest.approx(50.0)

    simile = _vector("他好像走。")
    assert simile["simile_density"] == pytest.approx(20.0)
    assert _vector("仿佛相似。")["simile_density"] == pytest.approx(200 / 5)

    function_words = _vector("他的书在桌上了。")
    assert function_words["fw_的"] == pytest.approx(12.5)
    assert function_words["fw_了"] == pytest.approx(12.5)
    assert function_words["fw_着"] == 0.0
    assert function_words["fw_地"] == 0.0
    assert function_words["fw_而"] == 0.0
    assert function_words["fw_其"] == 0.0
    assert function_words["fw_之"] == 0.0

    lexicon = LiteraryLexicon(literary={"仓库": 1.5}, colloquial={"很大": -1.5})
    literary = _vector("仓库很大。", lexicon)
    assert literary["literary_hit"] == pytest.approx(1 / 3)
    assert _vector("仓库很大。")["literary_hit"] == 0.0

    paragraphs = _vector("甲乙。\n\n丙丁。")
    assert paragraphs["para_density"] == pytest.approx(25.0)
    assert paragraphs["sent_len_mean"] == pytest.approx(2.0)


def test_cjk_numeral_ratio_ignores_digit_shape() -> None:
    assert _vector("第四天")["cjk_numeral_ratio"] == 1.0
    assert _vector("第４天")["cjk_numeral_ratio"] == 0.0
    assert _vector("第4天")["cjk_numeral_ratio"] == 0.0
    assert _vector("三4")["cjk_numeral_ratio"] == pytest.approx(0.5)
    assert _vector("没有数字")["cjk_numeral_ratio"] == 0.0


def test_degenerate_inputs_do_not_raise() -> None:
    empty = extract("", LiteraryLexicon())
    assert empty.shape == (20,)
    assert empty.dtype == np.float64
    assert np.all(empty == 0)
    punct = _vector("，。！")
    assert punct["comma_ratio"] == pytest.approx(1 / 3)
    assert punct["period_ratio"] == pytest.approx(1 / 3)
    assert punct["sent_len_mean"] == 0.0
    single = _vector("啊")
    assert single["sent_len_mean"] == 0.0
    assert single["ttr"] == 1.0
    assert single["para_density"] == pytest.approx(100.0)


def test_lexicon_thresholds_are_symmetric_and_strict() -> None:
    lexicon = build_lexicon(
        {"书面": 3, "两边": 3, "太少": 2, "没有": 0},
        {"口语": 3, "两边": 1, "口语少": 2, "没有": 1},
    )
    assert lexicon.literary["书面"] == pytest.approx(math.log(4))
    assert "两边" not in lexicon.literary
    assert "太少" not in lexicon.literary
    assert lexicon.colloquial["口语"] == pytest.approx(math.log(0.25))
    assert "口语少" not in lexicon.colloquial
    assert lexicon_payload(lexicon) == {
        "literary": {"书面": lexicon.literary["书面"]},
        "colloquial": {"口语": lexicon.colloquial["口语"]},
    }
    assert build_lexicon({}, {}).literary == {}


def test_distance_is_zero_on_the_fitted_text_and_positive_elsewhere() -> None:
    lexicon = LiteraryLexicon()
    reference = StyleReference.fit(["甲乙丙丁。戊己庚辛。", "甲乙丙丁。戊己庚辛。"], lexicon)
    assert reference.distance("甲乙丙丁。戊己庚辛。") == pytest.approx(0.0)
    assert np.all(reference.std == STD_FLOOR)
    assert reference.distance("他说：「好像走了。」") > 0
    with pytest.raises(ValueError, match="至少需要一条原文"):
        StyleReference.fit([], lexicon)


def test_public_originals_are_closer_than_their_vernaculars() -> None:
    rows = [
        json.loads(line)
        for line in Path("corpus/sample_public.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lexicon = LiteraryLexicon.model_validate_json(
        Path("stylometry/data/lexicon.json").read_text(encoding="utf-8")
    )
    payload = json.loads(Path("stylometry/data/style_reference.json").read_text(encoding="utf-8"))
    reference = StyleReference(
        mean=np.array(payload["mean"], dtype=np.float64),
        std=np.array(payload["std"], dtype=np.float64),
        lexicon=lexicon,
    )
    originals = [reference.distance(row["original"]) for row in rows]
    vernaculars = [reference.distance(row["vernacular"]) for row in rows]
    assert all(
        original < vernacular for original, vernacular in zip(originals, vernaculars, strict=True)
    )
    assert float(np.mean(originals)) < float(np.mean(vernaculars))
