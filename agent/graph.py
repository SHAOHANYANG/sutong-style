"""Assemble and run the self-check rewrite graph."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agent.config import AgentConfig
from agent.nodes import (
    Generator,
    MessageBuilder,
    Retriever,
    RouteFn,
    Scorer,
    Verifier,
    default_route,
    make_generate,
    make_retrieve,
    make_route_node,
    make_score,
    make_verify,
)
from agent.prompts import make_revision_message_builder
from agent.state import (
    MAX_GENERATIONS,
    RECURSION_LIMIT,
    AgentResult,
    AgentState,
    FallbackKind,
    RoundRecord,
    TerminationReason,
    TraceEvent,
)

NowFn = Callable[[], str]
MonoFn = Callable[[], float]


def _default_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _default_mono() -> float:
    return time.monotonic()


@dataclass
class _RunBox:
    """Side channel so a recursion-limit abort still has the last known state."""

    state: AgentState | None = None


def pick_best_index(
    candidates: list[str],
    violation_counts: list[int],
    scores: list[float],
) -> int | None:
    """Lexicographic pick: fewest violations, then highest score, then earlier round.

    Blank strings (empty or whitespace-only) do not participate.
    """
    best_index: int | None = None
    best_key: tuple[int, float, int] | None = None
    for index, text in enumerate(candidates):
        if not text.strip():
            continue
        key = (violation_counts[index], -scores[index], index)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def _classify_termination(state: AgentState) -> tuple[TerminationReason, FallbackKind | None]:
    if state.get("hit_recursion_limit"):
        return "recursion_limit", None
    halt_kind = state.get("halt_kind")
    if halt_kind is not None:
        return "fallback", halt_kind
    iterations = int(state.get("iter") or 0)
    limit = int(state.get("max_generations") or MAX_GENERATIONS)
    violations = state.get("last_violations") or []
    score = state.get("last_score")
    threshold = state.get("score_threshold")
    score_unfinished = threshold is not None and score is not None and score < float(threshold)
    unfinished = bool(violations) or score_unfinished
    if iterations >= limit and unfinished:
        return "max_rounds", None
    return "accepted", None


def _payload_for(node_name: str, before: AgentState, after: AgentState) -> dict[str, Any]:
    """Build a prose-free payload for one node visit."""
    if node_name == "retrieve":
        exemplars = after.get("exemplars") or []
        return {
            "exemplar_ids": [item.id for item in exemplars],
            "is_re_retrieve": int(before.get("iter") or 0) > 0,
        }
    if node_name == "generate":
        error_type = after.get("last_generate_error_type")
        return {
            "round": int(after.get("iter") or 0),
            "feedback_format": after.get("feedback_format") or before.get("feedback_format"),
            "prompt_sha256": after.get("last_prompt_sha256") or "",
            "output_sha256": after.get("last_output_sha256") or "",
            "output_chars": int(after.get("last_output_chars") or 0),
            "raised": error_type is not None,
            "error_type": error_type,
        }
    if node_name == "verify":
        violations = after.get("last_violations") or []
        return {
            "violation_count": len(violations),
            "violations": [
                {
                    "kind": item.kind,
                    "expected": item.expected,
                    "actual": item.actual,
                }
                for item in violations
            ],
        }
    if node_name == "score":
        return {"score": after.get("last_score")}
    if node_name == "route":
        return {"decision": after.get("last_route")}
    return {}


def build_result(state: AgentState, vernacular: str) -> AgentResult:
    """Never return empty success. Fallbacks label themselves."""
    candidates = list(state.get("candidates") or [])
    violation_counts = list(state.get("violation_counts") or [])
    violations_history = list(state.get("violations_history") or [])
    scores = list(state.get("scores") or [])
    while len(violation_counts) < len(candidates):
        violation_counts.append(0)
    while len(scores) < len(candidates):
        scores.append(float("-inf"))
    while len(violations_history) < len(candidates):
        violations_history.append([])

    rounds = [
        RoundRecord(
            round=index + 1,
            output=text,
            violation_count=violation_counts[index],
            violations=list(violations_history[index]),
            score=scores[index],
        )
        for index, text in enumerate(candidates)
    ]
    termination, fallback_kind = _classify_termination(state)
    total_rounds = int(state.get("iter") or 0)
    re_retrieved = bool(state.get("re_retrieved", False))
    trace = list(state.get("trace") or [])
    error = state.get("error")

    if fallback_kind == "first_round_error":
        return AgentResult(
            output=vernacular,
            selected_round=None,
            total_rounds=total_rounds,
            termination=termination,
            fallback_kind=fallback_kind,
            error=error,
            rounds=rounds,
            re_retrieved=re_retrieved,
            trace=trace,
        )

    if fallback_kind == "later_round_error":
        index = pick_best_index(candidates, violation_counts, scores)
        if index is None:
            return AgentResult(
                output=vernacular,
                selected_round=None,
                total_rounds=total_rounds,
                termination="fallback",
                fallback_kind="all_blank",
                error=error,
                rounds=rounds,
                re_retrieved=re_retrieved,
                trace=trace,
            )
        return AgentResult(
            output=candidates[index],
            selected_round=index + 1,
            total_rounds=total_rounds,
            termination=termination,
            fallback_kind=fallback_kind,
            error=error,
            rounds=rounds,
            re_retrieved=re_retrieved,
            trace=trace,
        )

    index = pick_best_index(candidates, violation_counts, scores)
    if index is None:
        return AgentResult(
            output=vernacular,
            selected_round=None,
            total_rounds=total_rounds,
            termination="fallback",
            fallback_kind="all_blank",
            error=error,
            rounds=rounds,
            re_retrieved=re_retrieved,
            trace=trace,
        )

    return AgentResult(
        output=candidates[index],
        selected_round=index + 1,
        total_rounds=total_rounds,
        termination=termination,
        fallback_kind=None,
        error=error,
        rounds=rounds,
        re_retrieved=re_retrieved,
        trace=trace,
    )


def build_graph(
    *,
    retriever: Retriever,
    generator: Generator,
    verifier: Verifier,
    scorer: Scorer,
    message_builder: MessageBuilder | None = None,
    route_fn: RouteFn | None = None,
    now_fn: NowFn | None = None,
    mono_fn: MonoFn | None = None,
) -> tuple[Any, _RunBox]:
    """Compile retrieve → generate → verify → score → route."""
    builder = message_builder
    choose = route_fn or default_route
    clock_now = now_fn or _default_now
    clock_mono = mono_fn or _default_mono
    box = _RunBox()
    if builder is None:
        raise ValueError("message_builder 必须由 run_agent 注入")

    def _watch(
        node_name: str, fn: Callable[[AgentState], AgentState]
    ) -> Callable[[AgentState], dict[str, Any]]:
        def wrapped(state: AgentState) -> dict[str, Any]:
            started = clock_mono()
            update = fn(state)
            duration_ms = (clock_mono() - started) * 1000.0
            merged: AgentState = {**state, **update}
            event = TraceEvent(
                node=node_name,
                ts=clock_now(),
                duration_ms=duration_ms,
                payload=_payload_for(node_name, state, merged),
            )
            new_trace = [*list(state.get("trace") or []), event]
            with_trace: dict[str, Any] = {**update, "trace": new_trace}
            box.state = {**merged, "trace": new_trace}
            return with_trace

        wrapped.__name__ = node_name
        return wrapped

    # langgraph's add_node overloads reject ordinary callables under mypy strict;
    # the runtime API accepts them. Keep the graph untyped at the builder boundary.
    graph: Any = StateGraph(AgentState)
    graph.add_node("retrieve", _watch("retrieve", make_retrieve(retriever)))
    graph.add_node("generate", _watch("generate", make_generate(generator, builder)))
    graph.add_node("verify", _watch("verify", make_verify(verifier)))
    graph.add_node("score", _watch("score", make_score(scorer)))
    graph.add_node("route", _watch("route", make_route_node(choose)))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "score")
    graph.add_edge("score", "route")
    graph.add_conditional_edges(
        "route",
        choose,
        {
            "accept": END,
            "revise": "generate",
            "re_retrieve": "retrieve",
        },
    )
    return graph.compile(), box


def run_agent(
    vernacular: str,
    *,
    retriever: Retriever,
    generator: Generator,
    verifier: Verifier,
    scorer: Scorer,
    config: AgentConfig | None = None,
    score_threshold: float | None = None,
    message_builder: MessageBuilder | None = None,
    route_fn: RouteFn | None = None,
    now_fn: NowFn | None = None,
    mono_fn: MonoFn | None = None,
) -> AgentResult:
    """Run the loop. Recursion-limit and generator failures still return a labeled result."""
    settings = (
        config if config is not None else AgentConfig(re_retrieve_score_threshold=score_threshold)
    )
    builder = message_builder or make_revision_message_builder(settings.feedback_format)
    compiled, box = build_graph(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        scorer=scorer,
        message_builder=builder,
        route_fn=route_fn,
        now_fn=now_fn,
        mono_fn=mono_fn,
    )
    initial: AgentState = {
        "input": vernacular,
        "exemplars": [],
        "candidates": [],
        "violation_counts": [],
        "violations_history": [],
        "scores": [],
        "last_violations": [],
        "iter": 0,
        "re_retrieved": False,
        "max_generations": settings.max_generations,
        "score_threshold": settings.re_retrieve_score_threshold,
        "feedback_format": settings.feedback_format,
        "trace": [],
        "halt": False,
    }
    try:
        final_raw = compiled.invoke(initial, {"recursion_limit": RECURSION_LIMIT})
        state = cast(AgentState, final_raw)
    except GraphRecursionError:
        state = cast(AgentState, dict(box.state or initial))
        state["hit_recursion_limit"] = True
    return build_result(state, vernacular)
