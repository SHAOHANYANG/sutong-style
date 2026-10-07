"""Deterministic fake implementations for CPU-only tests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence

import numpy as np
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from eval.fidelity import Violation
from retrieval.prompt import Exemplar
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


class FakePromptTokenizer:
    """Character count of message contents. No tokenizer and no download."""

    def count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


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


class FakeAgentRetriever:
    """Programmable exemplar lists, keyed by call index."""

    def __init__(self, batches: Sequence[Sequence[Exemplar]] | None = None) -> None:
        self.batches = [list(batch) for batch in (batches or [[]])]
        self.calls: list[str] = []
        self.node_calls = 0

    def retrieve(self, vernacular: str) -> list[Exemplar]:
        self.node_calls += 1
        self.calls.append(vernacular)
        index = min(len(self.calls) - 1, len(self.batches) - 1)
        return list(self.batches[index])


class FakeAgentGenerator:
    """Programmable chat completions. Call count and message args are recorded."""

    def __init__(
        self,
        outputs: Sequence[str] | None = None,
        *,
        errors_on: Sequence[int] | None = None,
    ) -> None:
        self.outputs = list(outputs) if outputs is not None else ["生成结果"]
        self.errors_on = set(errors_on or [])
        self.calls: list[list[dict[str, str]]] = []
        self.node_calls = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.node_calls += 1
        self.calls.append(list(messages))
        call_index = len(self.calls) - 1
        if call_index in self.errors_on:
            raise RuntimeError(f"fake generator failure on call {call_index}")
        if call_index >= len(self.outputs):
            return self.outputs[-1]
        return self.outputs[call_index]


class FakeAgentVerifier:
    """Programmable violation lists by call index."""

    def __init__(
        self,
        batches: Sequence[Sequence[Violation]] | None = None,
        *,
        always: Sequence[Violation] | None = None,
    ) -> None:
        if always is not None:
            self._always: list[Violation] | None = list(always)
            self.batches: list[list[Violation]] = []
        else:
            self._always = None
            self.batches = [list(batch) for batch in (batches or [[]])]
        self.calls: list[tuple[str, str]] = []
        self.node_calls = 0

    def verify(self, vernacular: str, output: str) -> list[Violation]:
        self.node_calls += 1
        self.calls.append((vernacular, output))
        if self._always is not None:
            return list(self._always)
        index = min(len(self.calls) - 1, len(self.batches) - 1)
        return list(self.batches[index])


class FakeAgentScorer:
    """Programmable scores by call index. Higher is better."""

    def __init__(
        self,
        values: Sequence[float] | None = None,
        *,
        always: float | None = None,
        fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.values = list(values) if values is not None else [1.0]
        self.always = always
        self.fn = fn
        self.calls: list[tuple[str, str]] = []
        self.node_calls = 0

    def score(self, vernacular: str, output: str) -> float:
        self.node_calls += 1
        self.calls.append((vernacular, output))
        if self.fn is not None:
            return float(self.fn(vernacular, output))
        if self.always is not None:
            return float(self.always)
        index = min(len(self.calls) - 1, len(self.values) - 1)
        return float(self.values[index])


class FakeStylePredictor:
    """Returns a fixed target vector. Used to unit-test PredictorScorer offline."""

    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.calls: list[np.ndarray] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(x, dtype=np.float64))
        return self.target.copy()
