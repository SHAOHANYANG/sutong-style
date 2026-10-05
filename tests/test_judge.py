import json
from pathlib import Path

import pytest

from eval.judge import DiskCache, cached_choice, compare_case, outcome_of, style_win_rate

PROMPT = Path("eval/prompts/pairwise_judge.txt").read_text(encoding="utf-8")


class AlwaysA:
    """Pick the text in slot A, whatever it says."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return "A"


def test_agreement_is_a_win_or_a_loss() -> None:
    assert outcome_of("A", "B") == "win"
    assert outcome_of("B", "A") == "loss"
    assert outcome_of("A", "A") == "tie"
    assert outcome_of("B", "B") == "tie"
    with pytest.raises(ValueError, match="没有可比的样本"):
        style_win_rate(())


class Garbage:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return "不知道"


def test_unparsed_vote_is_not_cached(tmp_path: Path) -> None:
    judge = Garbage()
    with pytest.raises(ValueError, match="A 或 B"):
        cached_choice(judge, DiskCache(tmp_path / "cache"), "提示", "fake-judge")
    assert judge.calls == 1
    assert not (tmp_path / "cache").exists()


def test_position_swap_pulls_an_always_a_judge_to_one_half(tmp_path: Path) -> None:
    judge = AlwaysA()
    cache = DiskCache(tmp_path / "cache")
    outcomes = [
        compare_case(judge, cache, PROMPT, "fake-judge", f"参照{i}", f"系统{i}", f"白话{i}")
        for i in range(4)
    ]
    rate = style_win_rate(outcomes)
    assert rate == 0.5
    assert rate != 1.0
    assert judge.calls == 8


def test_cache_hit_does_not_call_the_judge_again(tmp_path: Path) -> None:
    judge = AlwaysA()
    cache = DiskCache(tmp_path / "cache")
    first = compare_case(judge, cache, PROMPT, "fake-judge", "参照", "系统", "白话")
    calls_after_first = judge.calls
    second = compare_case(judge, cache, PROMPT, "fake-judge", "参照", "系统", "白话")
    assert first == second == "tie"
    assert calls_after_first == 2
    assert judge.calls == 2
    stored = list((tmp_path / "cache").iterdir())
    assert len(stored) == 2
    assert {path.read_text(encoding="utf-8").strip() for path in stored} == {"A"}
    assert json.dumps({"原文": "参照"}) not in stored[0].read_text(encoding="utf-8")
