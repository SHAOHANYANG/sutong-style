"""Few-shot messages in the training chat shape. The system line is not rewritten here."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from scripts.train import SYSTEM_PROMPT, build_messages

MAX_NEW_TOKENS = 768
MAX_SEQ_LENGTH = 4096


class Exemplar(BaseModel):
    """One training pair shown as its own user/assistant turn."""

    id: str
    vernacular: str
    original: str


class PromptTokenizer(Protocol):
    """Token count for a chat-template message list. Injected so tests stay offline."""

    def count(self, messages: list[dict[str, str]]) -> int: ...


def build_prompt(vernacular: str, exemplars: Sequence[Exemplar]) -> list[dict[str, str]]:
    """Rank 1 sits immediately before the real user turn. Empty exemplars match training."""
    if not exemplars:
        return build_messages(vernacular, None)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for exemplar in reversed(list(exemplars)):
        messages.append({"role": "user", "content": exemplar.vernacular})
        messages.append({"role": "assistant", "content": exemplar.original})
    messages.append({"role": "user", "content": vernacular})
    return messages


def assert_within_budget(
    sample_id: str,
    prompt_tokens: int,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> None:
    """Fail the run when the prompt plus the generation cap does not fit. Never truncate."""
    if prompt_tokens + max_new_tokens > max_seq_length:
        raise ValueError(
            f"样本 {sample_id}：prompt token 数 {prompt_tokens}"
            f" + max_new_tokens {max_new_tokens} 超过 max_seq_length {max_seq_length}"
        )
