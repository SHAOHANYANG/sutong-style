"""Deterministic fake implementations for CPU-only tests."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
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


class FakeTwoPassCall(BaseModel):
    stage: int
    seed: int
    temperature: float
    top_p: float
    max_tokens: int
    thinking_mode: str


class FakeTwoPassGenerator:
    def __init__(self, *, invalid_json: bool = False) -> None:
        self.calls: list[FakeTwoPassCall] = []
        self.invalid_json = invalid_json

    def generate(self, prompt: str, **kwargs: object) -> str:
        stage = 2 if prompt.startswith("这是第二遍") else 1
        self.calls.append(FakeTwoPassCall.model_validate({"stage": stage, **kwargs}))
        text = "甲乙在河边聊了很久今天要去镇上看看新房子" * 12
        if stage == 1:
            return text
        if self.invalid_json:
            return "非JSON，禁止重试"
        return json.dumps({"issues": [], "repaired_text": text}, ensure_ascii=False)


class FakeEmbedder:
    """Deterministic encoder. Injected vectors win; everything else is a text hash."""

    def __init__(self, *, vectors: dict[str, np.ndarray] | None = None, dim: int = 4) -> None:
        if dim < 1:
            raise ValueError("dim 必须是正整数")
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = {} if vectors is None else vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self._row(text) for text in texts])

    def _row(self, text: str) -> np.ndarray:
        if text in self.vectors:
            row = np.asarray(self.vectors[text], dtype=np.float64)
            if row.shape != (self.dim,):
                raise ValueError("FakeEmbedder 向量维度不一致")
            return row
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.dim)
