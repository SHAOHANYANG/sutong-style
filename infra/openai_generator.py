"""OpenAI-compatible text generator used only at command-line composition roots."""

from __future__ import annotations

import threading

from openai import OpenAI

from scripts.rebuild_reporting import ProviderMetadata


class OpenAICompatibleGenerator:
    """Generate text through an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=120.0)
        self._model = model
        self._local = threading.local()

    def response_metadata(self) -> ProviderMetadata:
        """Return the response identity for the current worker thread."""
        value: object = getattr(self._local, "metadata", None)
        return value if isinstance(value, ProviderMetadata) else ProviderMetadata()

    def generate(self, prompt: str, **kwargs: object) -> str:
        seed = kwargs.get("seed")
        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature) if isinstance(temperature, (float, int)) else 0.2,
            top_p=float(top_p) if isinstance(top_p, (float, int)) else 1.0,
            max_tokens=max_tokens if isinstance(max_tokens, int) else 2048,
            seed=seed if isinstance(seed, int) else None,
        )
        self._local.metadata = ProviderMetadata(
            model=response.model,
            response_id=response.id,
            created=response.created,
            system_fingerprint=response.system_fingerprint,
            finish_reason=response.choices[0].finish_reason,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM 返回了空内容")
        return content.strip()
