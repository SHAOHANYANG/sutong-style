"""Clients for the model server (scripts/serve_model.py, or any OpenAI-compatible host).

The API process uses these instead of loading a model: it must not import torch.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from retrieval.prompt import MAX_NEW_TOKENS

# The local server does not check keys; the SDK refuses an empty one.
PLACEHOLDER_KEY = "not-needed"


class ChatCompletionsGenerator:
    """Agent Generator over /v1/chat/completions. Decoding is greedy on the server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = PLACEHOLDER_KEY,
        max_new_tokens: int = MAX_NEW_TOKENS,
        timeout: float = 300.0,
        # An httpx-compatible client; tests pass an in-process one.
        http_client: Any = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        self._model = model
        self._max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self._complete(messages, None)

    def generate_with_seed(self, messages: list[dict[str, str]], *, seed: int) -> str:
        return self._complete(messages, seed)

    def ping(self) -> bool:
        return self._model in {item.id for item in self._client.models.list().data}

    def _complete(self, messages: list[dict[str, str]], seed: int | None) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            # The SDK types each role separately; the agent only sends system/user/assistant.
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=self._max_new_tokens,
            temperature=0.0,
            seed=seed,
        )
        content = response.choices[0].message.content if response.choices else None
        if content is None:
            raise RuntimeError("模型服务没有返回内容")
        return content


class RemoteEmbedder:
    """Query encoder over /v1/embeddings. The revision must match the document index."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_revision: str,
        api_key: str = PLACEHOLDER_KEY,
        timeout: float = 120.0,
        # An httpx-compatible client; tests pass an in-process one.
        http_client: Any = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        self._model = model
        self._expected_revision = expected_revision

    def served_revision(self) -> str | None:
        for item in self._client.models.list().data:
            if item.id == self._model:
                extra = item.model_extra or {}
                value = extra.get("revision")
                return value if isinstance(value, str) else None
        return None

    def ping(self) -> bool:
        # A different revision would silently compare vectors from two encoders.
        return self.served_revision() == self._expected_revision

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._model, input=texts, encoding_format="float"
        )
        rows = sorted(response.data, key=lambda item: item.index)
        return np.asarray([row.embedding for row in rows], dtype=np.float64)
