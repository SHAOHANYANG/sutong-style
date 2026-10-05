import math

from eval.fidelity import JiebaTagger, assess, build_entity_gazetteer, extract_facts
from eval.metrics import entity_recall, hallucination_rate, numeral_recall


class EmptyTagger:
    def entities(self, text: str) -> set[str]:
        return set()


def test_changed_magnitude_lowers_numeral_recall() -> None:
    report = assess("万人大军", "十万铁骑", tagger=EmptyTagger())
    assert report.numeral_recall < 1.0
    assert extract_facts("万人大军", set(), EmptyTagger()).numerals == {"10000"}
    assert "10000" not in extract_facts("十万铁骑", set(), EmptyTagger()).numerals


def test_swapped_title_is_a_violation() -> None:
    report = assess("太医来了", "宫监来了", tagger=EmptyTagger())
    assert len(report.violations) == 1
    assert report.violations[0].kind == "title"
    assert report.violations[0].expected == "太医"
    assert report.violations[0].actual == "宫监"


def test_semantic_paraphrase_does_not_crash() -> None:
    report = assess("垂死的酸气", "死尸散发的酸臭之气", tagger=EmptyTagger())
    assert math.isfinite(report.entity_recall)
    assert math.isfinite(report.numeral_recall)
    assert math.isfinite(report.hallucination_rate)
    assert report.violations == []


def test_fullwidth_digit_matches_chinese_numeral() -> None:
    report = assess("三辆马车", "３辆马车", tagger=EmptyTagger())
    assert report.numeral_recall == 1.0
    assert extract_facts("三辆马车", set(), EmptyTagger()).numerals == {"3"}


def test_cn2an_normalization_cases() -> None:
    assert extract_facts("十万人", set(), EmptyTagger()).numerals == {"100000"}
    assert extract_facts("万人", set(), EmptyTagger()).numerals == {"10000"}
    assert extract_facts("三月初九", set(), EmptyTagger()).numerals == {"3", "9"}
    assert extract_facts("四斤米", set(), EmptyTagger()).numerals == {"4"}
    assert extract_facts("二十人", set(), EmptyTagger()).numerals == {"20"}
    assert extract_facts("一百二十斤", set(), EmptyTagger()).numerals == {"120"}


def test_empty_sets_use_the_specified_boundaries() -> None:
    assert entity_recall(set(), {"颂莲"}) == 1.0
    assert numeral_recall(set(), {"3"}) == 1.0
    assert hallucination_rate({"颂莲"}, set()) == 0.0
    assert hallucination_rate(set(), set()) == 0.0
    empty_output = assess("颂莲来了", "", {"颂莲"}, tagger=EmptyTagger())
    assert empty_output.entity_recall == 0.0
    assert empty_output.hallucination_rate == 0.0
    assert empty_output.numeral_recall == 1.0


def test_gazetteer_hit_is_not_a_hallucination_when_shared() -> None:
    report = assess("颂莲坐着", "颂莲站着", {"颂莲"}, tagger=EmptyTagger())
    assert report.entity_recall == 1.0
    assert report.hallucination_rate == 0.0


def test_missing_title_without_a_replacement() -> None:
    report = assess("太医来了", "他来了", tagger=EmptyTagger())
    assert report.violations[0].actual is None


def test_extra_entity_is_a_hallucination() -> None:
    report = assess("屋里安静", "颂莲来了", {"颂莲"}, tagger=EmptyTagger())
    assert report.entity_recall == 1.0
    assert report.hallucination_rate == 1.0


def test_gazetteer_keeps_names_that_clear_the_count() -> None:
    class FixedTagger:
        def entities(self, text: str) -> set[str]:
            return {"沉草"} if "沉草" in text else set()

    assert build_entity_gazetteer(["沉草"] * 2, min_count=3, tagger=FixedTagger()) == set()
    assert build_entity_gazetteer(["沉草"] * 3, min_count=3, tagger=FixedTagger()) == {"沉草"}
    assert "北京" in JiebaTagger().entities("他住在北京。")
