"""Deterministic fake implementations for CPU-only tests."""

from __future__ import annotations

import re


class FakeGenerator:
    """Return the prompt's original body with deterministic wording changes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls.append(prompt)
        original = prompt.rsplit("原文：\n", maxsplit=1)[-1]
        return re.sub("想不到", "没想到", original).replace("甲", "乙")
