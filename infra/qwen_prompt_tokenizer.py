"""Count prompt tokens with the tokenizer the caller already loaded.

The count follows scripts.generate.generate_one: render the chat template to a
string, then tokenize that string with special tokens off. This module does not
import transformers and does not download a tokenizer of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class PromptTokenizer(Protocol):
    """The two calls generate_one makes while building the model input."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> str: ...

    def __call__(
        self, prompt: str, *, add_special_tokens: bool = ...
    ) -> Mapping[str, Sequence[int]]: ...


class QwenPromptTokenizer:
    """Token length of the string that would be fed to the model."""

    def __init__(self, tokenizer: PromptTokenizer) -> None:
        self._tokenizer = tokenizer

    def count(self, messages: list[dict[str, str]]) -> int:
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template 没有返回字符串")
        encoded = self._tokenizer(prompt, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        return len(input_ids)
