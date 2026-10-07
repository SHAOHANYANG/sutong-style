import os
import subprocess
import sys
from pathlib import Path

import pytest

from retrieval.prompt import (
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    Exemplar,
    assert_within_budget,
    build_prompt,
)
from scripts.train import SYSTEM_PROMPT, build_messages
from tests.fakes import FakePromptTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _exemplars() -> list[Exemplar]:
    return [
        Exemplar(id="r1", vernacular="张三去井边打水", original="井边站着一个人"),
        Exemplar(id="r2", vernacular="李四买了两袋米", original="米放在门口"),
        Exemplar(id="r3", vernacular="王五看着天", original="天色暗了下来"),
    ]


def test_zero_exemplars_match_the_training_messages() -> None:
    text = "张三去井边打水"
    assert build_prompt(text, []) == build_messages(text, None)


def test_exemplar_turns_keep_rank_one_next_to_the_input() -> None:
    exemplars = _exemplars()
    query = "真正的输入在这里"
    for k in (1, 2, 3):
        messages = build_prompt(query, exemplars[:k])
        assert len(messages) == 2 + 2 * k
        assert [message["role"] for message in messages] == (
            ["system"] + ["user", "assistant"] * k + ["user"]
        )
        assert messages[0]["content"] is SYSTEM_PROMPT
        assert messages[-1] == {"role": "user", "content": query}
        assert messages[-3]["content"] == exemplars[0].vernacular
        assert messages[-2]["content"] == exemplars[0].original
        assert messages[1]["content"] == exemplars[k - 1].vernacular
        allowed = {
            SYSTEM_PROMPT,
            query,
            *(item.vernacular for item in exemplars[:k]),
            *(item.original for item in exemplars[:k]),
        }
        assert {message["content"] for message in messages} == allowed
        assert all("范例" not in message["content"] for message in messages)


def test_fake_tokenizer_counts_characters() -> None:
    messages = build_prompt("张三", [])
    assert FakePromptTokenizer().count(messages) == len(SYSTEM_PROMPT) + len("张三")


def test_budget_accepts_equality_and_rejects_one_extra_token() -> None:
    room = MAX_SEQ_LENGTH - MAX_NEW_TOKENS
    assert_within_budget("case-9", room)
    with pytest.raises(ValueError, match="case-9") as caught:
        assert_within_budget("case-9", room + 1)
    text = str(caught.value)
    assert str(room + 1) in text
    assert str(MAX_NEW_TOKENS) in text
    assert str(MAX_SEQ_LENGTH) in text


def test_qwen_tokenizer_module_does_not_import_transformers() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import infra.qwen_prompt_tokenizer; import sys; "
            "assert 'transformers' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
