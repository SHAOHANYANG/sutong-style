"""Assemble and run the self-check rewrite graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    default_message_builder,
    default_route,
    make_generate,
    make_retrieve,
    make_route_node,
    make_score,
    make_verify,
)
from agent.state import (
    MAX_GENERATIONS,
    RECURSION_LIMIT,
    AgentResult,
    AgentState,
    FallbackKind,
    RoundRecord,
    TerminationReason,
)


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


def build_result(state: AgentState, vernacular: str) -> AgentResult:
    """Never return empty success. Fallbacks label themselves."""
    candidates = list(state.get("candidates") or [])
    violation_counts = list(state.get("violation_counts") or [])
    scores = list(state.get("scores") or [])
    while len(violation_counts) < len(candidates):
        violation_counts.append(0)
    while len(scores) < len(candidates):
        scores.append(float("-inf"))

    rounds = [
        RoundRecord(
            round=index + 1,
            output=text,
            violation_count=violation_counts[index],
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
) -> tuple[Any, _RunBox]:
    """Compile retrieve → generate → verify → score → route."""
    builder = message_builder or default_message_builder
    choose = route_fn or default_route
    box = _RunBox()

    def _watch(
        node_name: str, fn: Callable[[AgentState], AgentState]
    ) -> Callable[[AgentState], dict[str, Any]]:
        def wrapped(state: AgentState) -> dict[str, Any]:
            update = fn(state)
            merged: AgentState = {**state, **update}
            box.state = merged
            return dict(update)

        wrapped.__name__ = node_name
        return wrapped

    # langgraph's add_node overloads reject ordinary callables under mypy strict;
    # the runtime API accepts them. Keep the graph untyped at the builder boundary.
    graph: Any = StateGraph(AgentState)
    graph.add_node("retrieve", _watch("retrieve", make_retrieve(retriever)))
    graph.add_node("generate", _watch("generate", make_generate(generator, builder)))
    graph.add_node("verify", _watch("verify", make_verify(verifier)))
    graph.add_node("score", _watch("score", make_score(scorer)))
    graph.add_node("route", _watch("route", make_route_node()))

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
) -> AgentResult:
    """Run the loop. Recursion-limit and generator failures still return a labeled result."""
    settings = (
        config if config is not None else AgentConfig(re_retrieve_score_threshold=score_threshold)
    )
    compiled, box = build_graph(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        scorer=scorer,
        message_builder=message_builder,
        route_fn=route_fn,
    )
    initial: AgentState = {
        "input": vernacular,
        "exemplars": [],
        "candidates": [],
        "violation_counts": [],
        "scores": [],
        "last_violations": [],
        "iter": 0,
        "re_retrieved": False,
        "max_generations": settings.max_generations,
        "score_threshold": settings.re_retrieve_score_threshold,
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
