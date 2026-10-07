"""Agent skeleton tests. All Fake, no models, no network."""

from __future__ import annotations

import ast
from pathlib import Path

from agent.graph import build_result, pick_best_index, run_agent
from agent.nodes import RouteFn, default_route
from agent.state import (
    MAX_GENERATIONS,
    RECURSION_LIMIT,
    AgentResult,
    AgentState,
    RouteDecision,
    TraceEntry,
)
from eval.fidelity import Violation
from retrieval.prompt import Exemplar
from tests.fakes import (
    FakeAgentGenerator,
    FakeAgentRetriever,
    FakeAgentScorer,
    FakeAgentVerifier,
)

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
FORBIDDEN_ROOTS = {
    "langchain",
    "langchain_core",
    "torch",
    "transformers",
}
FORBIDDEN_MODULES = {"eval.judge"}
EX = Exemplar(id="ex1", vernacular="白话范例", original="原文范例")
VIOLATION = Violation(kind="title", expected="太医", actual="宫监")


def _run(
    *,
    outputs: list[str] | None = None,
    violations: list[list[Violation]] | None = None,
    always_violations: list[Violation] | None = None,
    scores: list[float] | None = None,
    always_score: float | None = None,
    score_threshold: float = 0.5,
    batches: list[list[Exemplar]] | None = None,
    errors_on: list[int] | None = None,
    route_fn: RouteFn | None = None,
) -> tuple[AgentResult, FakeAgentGenerator, FakeAgentRetriever]:
    generator = FakeAgentGenerator(outputs, errors_on=errors_on)
    retriever = FakeAgentRetriever(batches or [[EX]])
    if always_violations is not None:
        verifier = FakeAgentVerifier(always=always_violations)
    else:
        verifier = FakeAgentVerifier(violations)
    scorer = FakeAgentScorer(scores, always=always_score)
    result = run_agent(
        "输入白话",
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        scorer=scorer,
        score_threshold=score_threshold,
        route_fn=route_fn,
    )
    return result, generator, retriever


def test_one_shot_accept() -> None:
    result, generator, _ = _run(outputs=["好的一版"], scores=[0.9], score_threshold=0.5)
    assert len(generator.calls) == 1
    assert result.termination == "accepted"
    assert result.fallback_kind is None
    assert result.output == "好的一版"
    assert result.selected_round == 1
    assert result.total_rounds == 1


def test_max_rounds_always_violate() -> None:
    result, generator, _ = _run(
        outputs=["a", "b", "c"],
        always_violations=[VIOLATION],
        always_score=0.9,
    )
    assert len(generator.calls) == MAX_GENERATIONS
    assert result.output != ""
    assert result.termination == "max_rounds"
    assert result.total_rounds == 3


def test_second_round_fixes_violation() -> None:
    result, generator, _ = _run(
        outputs=["错版", "对版"],
        violations=[[VIOLATION], []],
        scores=[0.9, 0.8],
    )
    assert len(generator.calls) == 2
    assert result.output == "对版"
    assert result.selected_round == 2
    assert result.termination == "accepted"


def test_pick_prefers_fewer_violations_over_higher_score() -> None:
    result, _, _ = _run(
        outputs=["高分有错", "低分无错"],
        violations=[[VIOLATION], []],
        scores=[0.99, 0.1],
        score_threshold=0.0,
    )
    assert result.output == "低分无错"
    assert result.selected_round == 2


def test_pick_higher_score_when_violations_tie() -> None:
    index = pick_best_index(["甲", "乙"], [0, 0], [0.2, 0.8])
    assert index == 1


def test_pick_earlier_round_when_fully_tied() -> None:
    index = pick_best_index(["甲", "乙"], [1, 1], [0.5, 0.5])
    assert index == 0


def test_re_retrieve_once_only() -> None:
    result, generator, retriever = _run(
        outputs=["一", "二", "三"],
        violations=[[], [], []],
        scores=[0.1, 0.1, 0.1],
        score_threshold=0.5,
        batches=[[EX], [Exemplar(id="ex2", vernacular="b", original="B")]],
    )
    assert len(retriever.calls) == 2
    # First generate, then one re_retrieve + generate; score still low but no third pool.
    assert len(generator.calls) == 2
    assert result.re_retrieved is True
    assert result.termination == "accepted"


def test_mixed_revise_then_re_retrieve_caps_at_three() -> None:
    result, generator, retriever = _run(
        outputs=["v1", "v2", "v3"],
        violations=[[VIOLATION], [], []],
        scores=[0.9, 0.1, 0.1],
        score_threshold=0.5,
        batches=[[EX], [Exemplar(id="ex2", vernacular="b", original="B")]],
    )
    assert len(generator.calls) <= MAX_GENERATIONS
    assert len(generator.calls) == 3
    assert len(retriever.calls) == 2
    assert result.total_rounds == 3


def test_recursion_limit_with_bad_router() -> None:
    def always_revise(_state: AgentState) -> RouteDecision:
        return "revise"

    result, generator, _ = _run(
        outputs=["x"] * 30,
        always_violations=[],
        always_score=1.0,
        route_fn=always_revise,
    )
    assert result.termination == "recursion_limit"
    assert result.output.strip() != ""
    assert result.fallback_kind is None
    assert len(generator.calls) > MAX_GENERATIONS


def test_fallback_first_round_error() -> None:
    result, generator, _ = _run(outputs=["不会用到"], errors_on=[0])
    assert len(generator.calls) == 1
    assert result.termination == "fallback"
    assert result.fallback_kind == "first_round_error"
    assert result.output == "输入白话"
    assert result.selected_round is None
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_fallback_later_round_error() -> None:
    result, generator, _ = _run(
        outputs=["先出一版", "不会用到"],
        violations=[[VIOLATION]],
        scores=[0.4],
        errors_on=[1],
    )
    assert len(generator.calls) == 2
    assert result.termination == "fallback"
    assert result.fallback_kind == "later_round_error"
    assert result.output == "先出一版"
    assert result.selected_round == 1
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_blank_then_good_revises() -> None:
    result, generator, _ = _run(
        outputs=["   ", "可用"],
        violations=[[]],
        scores=[0.9],
        score_threshold=0.5,
    )
    assert len(generator.calls) == 2
    assert result.output == "可用"
    assert result.selected_round == 2
    assert result.termination == "accepted"
    assert result.fallback_kind is None
    assert len(result.rounds) == 2
    assert result.rounds[0].output.strip() == ""


def test_fallback_all_blank() -> None:
    result, generator, _ = _run(
        outputs=["   ", "\n\t", ""],
        violations=[],
        scores=[],
        score_threshold=0.5,
    )
    assert len(generator.calls) == 3
    assert result.termination == "fallback"
    assert result.fallback_kind == "all_blank"
    assert result.output == "输入白话"
    assert result.selected_round is None
    assert len(result.rounds) == 3


def test_blank_output_not_selected_when_other_exists() -> None:
    result, _, _ = _run(
        outputs=["   ", "可用"],
        violations=[[]],
        scores=[0.1],
        score_threshold=0.0,
    )
    assert result.output == "可用"
    assert result.selected_round == 2
    assert result.fallback_kind is None


def test_trace_node_order_matches_execution() -> None:
    result, _, _ = _run(outputs=["ok"], scores=[0.9], score_threshold=0.5)
    nodes = [entry.node for entry in result.trace]
    assert nodes == ["retrieve", "generate", "verify", "score", "route"]


def test_trace_on_revise_path() -> None:
    result, _, _ = _run(
        outputs=["a", "b"],
        violations=[[VIOLATION], []],
        scores=[0.9, 0.9],
    )
    nodes = [entry.node for entry in result.trace]
    assert nodes[:5] == ["retrieve", "generate", "verify", "score", "route"]
    assert nodes[5:9] == ["generate", "verify", "score", "route"]


def test_import_ban_in_agent_package() -> None:
    offenders: list[str] = []
    for path in sorted(AGENT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in FORBIDDEN_ROOTS or alias.name in FORBIDDEN_MODULES:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                if root in FORBIDDEN_ROOTS or module in FORBIDDEN_MODULES:
                    offenders.append(f"{path.name}: from {module}")
    assert offenders == []


def test_deterministic_same_fakes() -> None:
    first, _, _ = _run(
        outputs=["一", "二"],
        violations=[[VIOLATION], []],
        scores=[0.2, 0.3],
        score_threshold=0.0,
    )
    second, _, _ = _run(
        outputs=["一", "二"],
        violations=[[VIOLATION], []],
        scores=[0.2, 0.3],
        score_threshold=0.0,
    )
    assert first == second


def test_recursion_limit_constant_matches_formula() -> None:
    # 1 retrieve + 3*(gen+ver+score+route) + 1 extra retrieve + margin 6
    assert RECURSION_LIMIT == 1 + 3 * 4 + 1 + 6


def test_default_route_accepts_at_cap_with_violations() -> None:
    state: AgentState = {
        "iter": MAX_GENERATIONS,
        "last_violations": [VIOLATION],
        "last_score": 0.9,
        "score_threshold": 0.5,
        "re_retrieved": False,
    }
    assert default_route(state) == "accept"


def test_build_result_preserves_trace_entries() -> None:
    state: AgentState = {
        "input": "白话",
        "candidates": ["出"],
        "violation_counts": [0],
        "scores": [0.7],
        "last_violations": [],
        "last_score": 0.7,
        "iter": 1,
        "score_threshold": 0.5,
        "re_retrieved": False,
        "trace": [TraceEntry(node="retrieve", round=0)],
    }
    result = build_result(state, "白话")
    assert result.trace[0].node == "retrieve"
