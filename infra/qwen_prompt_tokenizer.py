"""Qwen chat-template token counts. transformers is imported only when constructed."""

from __future__ import annotations

from typing import Any

from scripts.train import BASE_MODEL


class QwenPromptTokenizer:
    """Count the tokens generate.py will actually feed the model."""

    def __init__(self, model_id: str = BASE_MODEL) -> None:
        from transformers import AutoTokenizer

        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_id)

    def count(self, messages: list[dict[str, str]]) -> int:
        encoded: Any = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(encoded, list):
            raise TypeError("chat template 没有返回 token id 列表")
        return len(encoded)
