"""OpenAI-compatible text generator used only at command-line composition roots."""

from __future__ import annotations

from openai import OpenAI


class OpenAICompatibleGenerator:
    """Generate text through an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(self, prompt: str, **kwargs: object) -> str:
        seed = kwargs.get("seed")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            seed=seed if isinstance(seed, int) else None,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM 返回了空内容")
        return content.strip()
