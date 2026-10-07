"""Revision prompt tests for T2.3."""

from __future__ import annotations

from agent.config import AgentConfig
from agent.graph import run_agent
from agent.prompts import (
    FOLLOWUP_LEAD,
    RESTATE_LEAD,
    build_revision_messages,
    feedback_lines,
    format_violation,
    ordered_violations,
)
from eval.fidelity import Violation
from retrieval.prompt import Exemplar, build_prompt
from tests.fakes import (
    FakeAgentGenerator,
    FakeAgentRetriever,
    FakeAgentScorer,
    FakeAgentVerifier,
)

BANNED = ("忠实", "原意")
EX = Exemplar(id="ex1", vernacular="范例白话", original="范例原文")


def test_four_kinds_contain_fragments_and_ban_vague_words() -> None:
    cases = [
        Violation(kind="entity_missing", expected="颂莲", actual=None),
        Violation(kind="numeral_missing", expected="十二块", actual=None),
        Violation(kind="entity_hallucination", expected=None, actual="陈佐千"),
        Violation(kind="title", expected="太医", actual="宫监"),
    ]
    for item in cases:
        line = format_violation(item)
        for banned in BANNED:
            assert banned not in line
        if item.expected:
            assert f"「{item.expected}」" in line
        if item.actual:
            assert f"「{item.actual}」" in line
    numeral = format_violation(cases[1])
    assert "数值不变" in numeral
    assert "中文数字" in numeral


def test_title_missing_actual() -> None:
    line = format_violation(Violation(kind="title", expected="太医", actual=None))
    assert "「太医」" in line
    assert "宫监" not in line
    assert "不见了" in line


def test_order_and_dedupe() -> None:
    shuffled = [
        Violation(kind="title", expected="太医", actual="宫监"),
        Violation(kind="entity_missing", expected="颂莲", actual=None),
        Violation(kind="title", expected="太医", actual="宫监"),
        Violation(kind="numeral_missing", expected="十二块", actual=None),
    ]
    ordered = ordered_violations(shuffled)
    assert [item.kind for item in ordered] == [
        "entity_missing",
        "numeral_missing",
        "title",
    ]
    assert feedback_lines(shuffled) == feedback_lines(list(reversed(shuffled)))


def test_followup_structure_matches_build_prompt_prefix() -> None:
    vernacular = "太医来了"
    violations = [Violation(kind="title", expected="太医", actual="宫监")]
    messages = build_revision_messages(
        vernacular,
        [EX],
        previous_output="宫监来了",
        violations=violations,
        feedback_format="followup",
    )
    base = build_prompt(vernacular, [EX])
    assert messages[: len(base)] == base
    assert messages[-2] == {"role": "assistant", "content": "宫监来了"}
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith(FOLLOWUP_LEAD)
    assert "「太医」" in messages[-1]["content"]
    assert "「宫监」" in messages[-1]["content"]
    assert EX.vernacular in messages[1]["content"]
    assert EX.original in messages[2]["content"]


def test_restate_structure() -> None:
    vernacular = "输入白话"
    violations = [Violation(kind="entity_missing", expected="颂莲", actual=None)]
    messages = build_revision_messages(
        vernacular,
        [EX],
        previous_output="会被丢掉的上一版",
        violations=violations,
        feedback_format="restate",
    )
    base = build_prompt(vernacular, [EX])
    assert messages[:-1] == base[:-1]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith(vernacular + "\n\n" + RESTATE_LEAD)
    assert "会被丢掉的上一版" not in messages[-1]["content"]
    assert "「颂莲」" in messages[-1]["content"]


def test_followup_third_round_only_latest_output() -> None:
    vernacular = "输入"
    first = build_revision_messages(
        vernacular,
        [],
        previous_output="第一版输出",
        violations=[Violation(kind="entity_missing", expected="甲", actual=None)],
        feedback_format="followup",
    )
    second = build_revision_messages(
        vernacular,
        [],
        previous_output="第二版输出",
        violations=[Violation(kind="entity_missing", expected="乙", actual=None)],
        feedback_format="followup",
    )
    assert "第一版输出" in first[-2]["content"]
    assert "第二版输出" in second[-2]["content"]
    assert "第一版输出" not in second[-2]["content"]
    assert "「乙」" in second[-1]["content"]
    assert "「甲」" not in second[-1]["content"]


def test_end_to_end_second_call_contains_fragment() -> None:
    generator = FakeAgentGenerator(["宫监来了", "太医来了"])
    verifier = FakeAgentVerifier(
        [
            [Violation(kind="title", expected="太医", actual="宫监")],
            [],
        ]
    )
    run_agent(
        "太医来了",
        retriever=FakeAgentRetriever([[]]),
        generator=generator,
        verifier=verifier,
        scorer=FakeAgentScorer(always=0.0),
        config=AgentConfig(feedback_format="followup"),
    )
    assert len(generator.calls) == 2
    second = "\n".join(message["content"] for message in generator.calls[1])
    assert "「太医」" in second
    assert "「宫监」" in second
    for banned in BANNED:
        assert banned not in second
