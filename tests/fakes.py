"""Deterministic fake implementations for CPU-only tests."""

from __future__ import annotations

import re

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from scripts.rebuild_reporting import ModelIdentityMismatchError, ProviderMetadata


class FakeGenerator:
    """Return the prompt's original body with deterministic wording changes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls.append(prompt)
        original = prompt.rsplit("原文：\n", maxsplit=1)[-1]
        return re.sub("想不到", "没想到", original).replace("甲", "乙")


class FakeMismatchGenerator(FakeGenerator):
    """Simulate a provider routing error without any network access."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls.append(prompt)
        raise ModelIdentityMismatchError("requested", "returned")

    def response_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(model="returned", response_id="fake-routing-error")


class FakeProviderCall(BaseModel):
    model: str
    messages: list[dict[str, str]]
    seed: int | None
    temperature: float
    top_p: float
    max_tokens: int
    extra_body: dict[str, dict[str, str]] | None


class FakeCompletions:
    def __init__(self, model: str, requests: list[FakeProviderCall]) -> None:
        self.model = model
        self.requests = requests

    def create(self, **kwargs: object) -> ChatCompletion:
        self.requests.append(FakeProviderCall.model_validate(kwargs))
        return ChatCompletion.model_validate(
            {
                "id": "fake-response",
                "object": "chat.completion",
                "created": 1,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "假数据"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


class FakeChat:
    def __init__(self, model: str, requests: list[FakeProviderCall]) -> None:
        self.completions = FakeCompletions(model, requests)


class FakeOpenAI:
    """Pure SDK-shaped fake; no network, dependencies, or credentials."""

    def __init__(self, model: str, requests: list[FakeProviderCall]) -> None:
        self.chat = FakeChat(model, requests)
