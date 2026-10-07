"""Agent skeleton and TraceEvent tests. All Fake, no models, no network."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

from agent.graph import build_result, pick_best_index, run_agent
from agent.nodes import RouteFn, default_route, sha256_text
from agent.state import (
    MAX_GENERATIONS,
    RECURSION_LIMIT,
    AgentResult,
    AgentState,
    RouteDecision,
    TraceEvent,
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
FIXED_TS = "2026-10-07T12:00:00Z"


class FixedClock:
    """Deterministic wall clock and monotonic timer for identical traces."""

    def __init__(self, ts: str = FIXED_TS) -> None:
        self.ts = ts
        self._mono = 0.0

    def now(self) -> str:
        return self.ts

    def mono(self) -> float:
        value = self._mono
        self._mono += 0.001
        return value


def _run(
    *,
    outputs: list[str] | None = None,
    violations: list[list[Violation]] | None = None,
    always_violations: list[Violation] | None = None,
    scores: list[float] | None = None,
    always_score: float | None = None,
    score_threshold: float | None = None,
    batches: list[list[Exemplar]] | None = None,
    errors_on: list[int] | None = None,
    route_fn: RouteFn | None = None,
    vernacular: str = "输入白话",
    now_fn: FixedClock | None = None,
) -> tuple[AgentResult, FakeAgentGenerator, FakeAgentRetriever, FakeAgentVerifier, FakeAgentScorer]:
    clock = now_fn or FixedClock()
    generator = FakeAgentGenerator(outputs, errors_on=errors_on)
    retriever = FakeAgentRetriever(batches or [[EX]])
    if always_violations is not None:
        verifier = FakeAgentVerifier(always=always_violations)
    else:
        verifier = FakeAgentVerifier(violations)
    scorer = FakeAgentScorer(scores, always=always_score)
    result = run_agent(
        vernacular,
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        scorer=scorer,
        score_threshold=score_threshold,
        route_fn=route_fn,
        now_fn=clock.now,
        mono_fn=clock.mono,
    )
    return result, generator, retriever, verifier, scorer


def test_one_shot_accept() -> None:
    result, generator, _, _, _ = _run(outputs=["好的一版"], scores=[0.9])
    assert len(generator.calls) == 1
    assert result.termination == "accepted"
    assert result.fallback_kind is None
    assert result.output == "好的一版"
    assert result.selected_round == 1
    assert result.total_rounds == 1


def test_max_rounds_always_violate() -> None:
    result, generator, _, _, _ = _run(
        outputs=["a", "b", "c"],
        always_violations=[VIOLATION],
        always_score=0.9,
    )
    assert len(generator.calls) == MAX_GENERATIONS
    assert result.output != ""
    assert result.termination == "max_rounds"
    assert result.total_rounds == 3


def test_second_round_fixes_violation() -> None:
    result, generator, _, _, _ = _run(
        outputs=["错版", "对版"],
        violations=[[VIOLATION], []],
        scores=[0.9, 0.8],
    )
    assert len(generator.calls) == 2
    assert result.output == "对版"
    assert result.selected_round == 2
    assert result.termination == "accepted"
    assert result.rounds[0].violations == [VIOLATION]
    assert result.rounds[1].violations == []


def test_pick_prefers_fewer_violations_over_higher_score() -> None:
    result, _, _, _, _ = _run(
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
    result, generator, retriever, _, _ = _run(
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
    result, generator, retriever, _, _ = _run(
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

    result, generator, _, _, _ = _run(
        outputs=["x"] * 30,
        always_violations=[],
        always_score=1.0,
        route_fn=always_revise,
    )
    assert result.termination == "recursion_limit"
    assert result.output.strip() != ""
    assert result.fallback_kind is None
    assert len(generator.calls) > MAX_GENERATIONS
    assert len(result.trace) > 0
    assert result.trace[-1].node in {"retrieve", "generate", "verify", "score", "route"}


def test_fallback_first_round_error() -> None:
    result, generator, _, _, _ = _run(outputs=["不会用到"], errors_on=[0])
    assert len(generator.calls) == 1
    assert result.termination == "fallback"
    assert result.fallback_kind == "first_round_error"
    assert result.output == "输入白话"
    assert result.selected_round is None
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_fallback_later_round_error() -> None:
    result, generator, _, _, _ = _run(
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
    result, generator, _, _, _ = _run(
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
    result, generator, _, _, _ = _run(
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
    result, _, _, _, _ = _run(
        outputs=["   ", "可用"],
        violations=[[]],
        scores=[0.1],
        score_threshold=0.0,
    )
    assert result.output == "可用"
    assert result.selected_round == 2
    assert result.fallback_kind is None


def test_trace_node_order_matches_execution() -> None:
    result, _, _, _, _ = _run(outputs=["ok"], scores=[0.9], score_threshold=0.5)
    nodes = [entry.node for entry in result.trace]
    assert nodes == ["retrieve", "generate", "verify", "score", "route"]


def test_trace_on_revise_path() -> None:
    result, _, _, _, _ = _run(
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
    first, _, _, _, _ = _run(
        outputs=["一", "二"],
        violations=[[VIOLATION], []],
        scores=[0.2, 0.3],
        score_threshold=0.0,
    )
    second, _, _, _, _ = _run(
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
        "max_generations": MAX_GENERATIONS,
        "last_violations": [VIOLATION],
        "last_score": 0.9,
        "score_threshold": 0.5,
        "re_retrieved": False,
    }
    assert default_route(state) == "accept"


def test_null_threshold_skips_re_retrieve() -> None:
    result, generator, retriever, _, _ = _run(
        outputs=["低分"],
        violations=[[]],
        scores=[0.1],
        score_threshold=None,
    )
    assert len(generator.calls) == 1
    assert len(retriever.calls) == 1
    assert result.termination == "accepted"
    assert result.re_retrieved is False


def test_build_result_preserves_trace_entries() -> None:
    state: AgentState = {
        "input": "白话",
        "candidates": ["出"],
        "violation_counts": [0],
        "violations_history": [[]],
        "scores": [0.7],
        "last_violations": [],
        "last_score": 0.7,
        "iter": 1,
        "max_generations": MAX_GENERATIONS,
        "score_threshold": None,
        "re_retrieved": False,
        "trace": [
            TraceEvent(
                node="retrieve",
                ts=FIXED_TS,
                duration_ms=1.0,
                payload={"exemplar_ids": [], "is_re_retrieve": False},
            )
        ],
    }
    result = build_result(state, "白话")
    assert result.trace[0].node == "retrieve"
    assert result.trace[0].ts == FIXED_TS


def test_trace_json_roundtrip() -> None:
    result, _, _, _, _ = _run(
        outputs=["错版", "对版"],
        violations=[[VIOLATION], []],
        scores=[0.5, 0.6],
    )
    raw = result.model_dump_json()
    restored = AgentResult.model_validate_json(raw)
    assert restored.trace == result.trace
    as_list = json.loads(raw)["trace"]
    again = [TraceEvent.model_validate(item) for item in as_list]
    assert again == result.trace


def test_trace_event_count_one_shot() -> None:
    result, generator, retriever, verifier, scorer = _run(
        outputs=["好"],
        scores=[0.9],
    )
    counts = Counter(event.node for event in result.trace)
    assert counts == {
        "retrieve": 1,
        "generate": 1,
        "verify": 1,
        "score": 1,
        "route": 1,
    }
    assert len(result.trace) == 5
    assert retriever.node_calls == 1
    assert generator.node_calls == 1
    assert verifier.node_calls == 1
    assert scorer.node_calls == 1


def test_trace_event_count_max_rounds() -> None:
    result, generator, retriever, verifier, scorer = _run(
        outputs=["a", "b", "c"],
        always_violations=[VIOLATION],
        always_score=0.9,
    )
    counts = Counter(event.node for event in result.trace)
    assert counts["retrieve"] == 1
    assert counts["generate"] == 3
    assert counts["verify"] == 3
    assert counts["score"] == 3
    assert counts["route"] == 3
    assert len(result.trace) == 13
    assert generator.node_calls == 3
    assert verifier.node_calls == 3
    assert scorer.node_calls == 3
    assert retriever.node_calls == 1


def test_trace_event_count_mid_error() -> None:
    result, generator, retriever, verifier, scorer = _run(
        outputs=["先出一版", "不会用到"],
        violations=[[VIOLATION]],
        scores=[0.4],
        errors_on=[1],
    )
    counts = Counter(event.node for event in result.trace)
    # retrieve + round1 (4) + round2 generate (halt) + verify/score no-op + route
    assert counts["retrieve"] == 1
    assert counts["generate"] == 2
    assert counts["verify"] == 2
    assert counts["score"] == 2
    assert counts["route"] == 2
    assert len(result.trace) == 9
    assert generator.node_calls == 2
    # second verify/score skip protocol because halt
    assert verifier.node_calls == 1
    assert scorer.node_calls == 1
    assert retriever.node_calls == 1
    gen_events = [event for event in result.trace if event.node == "generate"]
    assert gen_events[1].payload["raised"] is True
    assert gen_events[1].payload["error_type"] == "RuntimeError"


def test_trace_payload_fields() -> None:
    result, _, _, _, _ = _run(
        outputs=["错版", "对版"],
        violations=[[VIOLATION], []],
        scores=[0.9, 0.8],
        batches=[[EX]],
    )
    by_node: dict[str, list[TraceEvent]] = {}
    for event in result.trace:
        by_node.setdefault(event.node, []).append(event)

    retrieve = by_node["retrieve"][0]
    assert retrieve.payload["exemplar_ids"] == ["ex1"]
    assert retrieve.payload["is_re_retrieve"] is False

    gen0 = by_node["generate"][0]
    assert gen0.payload["round"] == 1
    assert gen0.payload["feedback_format"] == "followup"
    assert isinstance(gen0.payload["prompt_sha256"], str)
    assert len(gen0.payload["prompt_sha256"]) == 64
    assert gen0.payload["output_sha256"] == sha256_text("错版")
    assert gen0.payload["output_chars"] == 2
    assert gen0.payload["raised"] is False
    assert gen0.payload["error_type"] is None

    verify0 = by_node["verify"][0]
    assert verify0.payload["violation_count"] == 1
    assert verify0.payload["violations"] == [
        {"kind": "title", "expected": "太医", "actual": "宫监"}
    ]

    score0 = by_node["score"][0]
    assert score0.payload["score"] == 0.9

    route0 = by_node["route"][0]
    assert route0.payload["decision"] == "revise"
    route1 = by_node["route"][1]
    assert route1.payload["decision"] == "accept"


def test_trace_identical_with_fixed_clock() -> None:
    clock_a = FixedClock()
    clock_b = FixedClock()
    first, _, _, _, _ = _run(
        outputs=["一", "二"],
        violations=[[VIOLATION], []],
        scores=[0.2, 0.3],
        now_fn=clock_a,
    )
    second, _, _, _, _ = _run(
        outputs=["一", "二"],
        violations=[[VIOLATION], []],
        scores=[0.2, 0.3],
        now_fn=clock_b,
    )
    assert first.trace == second.trace


def test_trace_payload_has_no_prose_bodies() -> None:
    marker_in = "MARKER_INPUT_XYZ99"
    marker_ex_v = "MARKER_EX_VERN_ABC"
    marker_ex_o = "MARKER_EX_ORIG_DEF"
    marker_out = "MARKER_OUTPUT_GHI"
    ex = Exemplar(id="ex_mark", vernacular=marker_ex_v, original=marker_ex_o)
    result, _, _, _, _ = _run(
        vernacular=marker_in,
        outputs=[marker_out],
        scores=[0.9],
        batches=[[ex]],
    )
    blob = json.dumps(
        [event.model_dump() for event in result.trace],
        ensure_ascii=False,
    )
    for marker in (marker_in, marker_ex_v, marker_ex_o, marker_out):
        assert marker not in blob
    # short fragments in violations are allowed; this path has none
    assert "ex_mark" in blob
    assert sha256_text(marker_out) in blob


def test_trace_complete_on_recursion_limit() -> None:
    def always_revise(_state: AgentState) -> RouteDecision:
        return "revise"

    result, _, _, _, _ = _run(
        outputs=["x"] * 40,
        always_violations=[],
        always_score=1.0,
        route_fn=always_revise,
    )
    assert result.termination == "recursion_limit"
    assert len(result.trace) >= RECURSION_LIMIT - 1
    nodes = {event.node for event in result.trace}
    assert nodes <= {"retrieve", "generate", "verify", "score", "route"}
    assert result.trace[-1].ts == FIXED_TS
    assert "duration_ms" in result.trace[-1].model_dump()


def test_re_retrieve_payload_flag() -> None:
    result, _, _, _, _ = _run(
        outputs=["一", "二"],
        violations=[[], []],
        scores=[0.1, 0.1],
        score_threshold=0.5,
        batches=[[EX], [Exemplar(id="ex2", vernacular="b", original="B")]],
    )
    retrieves = [event for event in result.trace if event.node == "retrieve"]
    assert len(retrieves) == 2
    assert retrieves[0].payload["is_re_retrieve"] is False
    assert retrieves[1].payload["is_re_retrieve"] is True
    assert retrieves[1].payload["exemplar_ids"] == ["ex2"]
