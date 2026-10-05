"""Pairwise style judge. Each case is asked twice with the sides swapped."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

Choice = Literal["A", "B"]
Outcome = Literal["win", "loss", "tie"]
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_MAX_TOKENS = 16
JUDGE_SEED = 42


class Completer(Protocol):
    """One prompt in, one completion out. Tests supply a fake."""

    def generate(self, prompt: str, **kwargs: object) -> str: ...


def render_prompt(template: str, original: str, text_a: str, text_b: str) -> str:
    """Fill the committed template. Candidate text is inserted literally."""
    return (
        template.replace("{{original}}", original)
        .replace("{{text_a}}", text_a)
        .replace("{{text_b}}", text_b)
    )


def parse_choice(text: str) -> Choice:
    """Accept only a bare A or B. Anything else is a failed call, not a vote."""
    token = text.strip().splitlines()[0].strip().rstrip("。．.") if text.strip() else ""
    if token == "A":
        return "A"
    if token == "B":
        return "B"
    preview = text.strip().replace("\n", " ")[:20]
    raise ValueError(f"评委没有只返回 A 或 B：{preview}")


def cache_key(prompt: str, model: str) -> str:
    """SPEC 3.3: sha256 of the prompt concatenated with the model name."""
    return hashlib.sha256((prompt + model).encode("utf-8")).hexdigest()


class DiskCache:
    """One file per prompt. The file stores the letter, not the prompt text."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, prompt: str, model: str) -> Choice | None:
        path = self.directory / cache_key(prompt, model)
        if not path.exists():
            return None
        cached = path.read_text(encoding="utf-8").strip()
        if cached == "A":
            return "A"
        if cached == "B":
            return "B"
        return None

    def put(self, prompt: str, model: str, choice: Choice) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / cache_key(prompt, model)).write_text(choice + "\n", encoding="utf-8")


def cached_choice(completer: Completer, cache: DiskCache, prompt: str, model: str) -> Choice:
    """Return a cached letter, or call the completer once and store the letter."""
    cached = cache.get(prompt, model)
    if cached is not None:
        return cached
    raw = completer.generate(
        prompt,
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        max_tokens=JUDGE_MAX_TOKENS,
        seed=JUDGE_SEED,
        thinking_mode="disabled",
    )
    choice = parse_choice(raw)
    cache.put(prompt, model, choice)
    return choice


def outcome_of(first: Choice, second: Choice) -> Outcome:
    """first is the vote when A is the system. second is the vote after the swap."""
    system_first = first == "A"
    system_second = second == "B"
    if system_first and system_second:
        return "win"
    if not system_first and not system_second:
        return "loss"
    return "tie"


def style_win_rate(outcomes: Sequence[Outcome]) -> float:
    """(wins + half the ties) / cases. An empty list is not a rate."""
    if not outcomes:
        raise ValueError("没有可比的样本")
    score = sum(1.0 if item == "win" else 0.5 if item == "tie" else 0.0 for item in outcomes)
    return score / len(outcomes)


def vote_tally(outcomes: Sequence[Outcome]) -> dict[str, float | int]:
    """Win, loss, and tie counts. tie_rate diagnoses position inconsistency."""
    if not outcomes:
        raise ValueError("没有可比的样本")
    win = sum(item == "win" for item in outcomes)
    loss = sum(item == "loss" for item in outcomes)
    tie = sum(item == "tie" for item in outcomes)
    n = len(outcomes)
    return {"win": win, "loss": loss, "tie": tie, "tie_rate": tie / n, "n": n}


def compare_case(
    completer: Completer,
    cache: DiskCache,
    template: str,
    model: str,
    original: str,
    system: str,
    opponent: str,
) -> Outcome:
    """Ask twice. Agreement is a win or a loss; disagreement is a tie."""
    forward = cached_choice(
        completer,
        cache,
        render_prompt(template, original, system, opponent),
        model,
    )
    swapped = cached_choice(
        completer,
        cache,
        render_prompt(template, original, opponent, system),
        model,
    )
    return outcome_of(forward, swapped)
