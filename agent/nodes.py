"""Node callables and protocols. No I/O; callers inject every external capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import structlog

from agent.state import (
    MAX_GENERATIONS,
    AgentState,
    FallbackKind,
    RouteDecision,
    TraceEntry,
)
from eval.fidelity import Violation
from retrieval.prompt import Exemplar, build_prompt

LOGGER = structlog.get_logger()


class Retriever(Protocol):
    """Return exemplars for a vernacular query."""

    def retrieve(self, vernacular: str) -> Sequence[Exemplar]: ...


class Generator(Protocol):
    """One completion for a chat message list."""

    def generate(self, messages: list[dict[str, str]]) -> str: ...


class Verifier(Protocol):
    """Deterministic fidelity check. Returns concrete violations."""

    def verify(self, vernacular: str, output: str) -> list[Violation]: ...


class Scorer(Protocol):
    """Style (or other) score. Higher is better. Sees the vernacular input."""

    def score(self, vernacular: str, output: str) -> float: ...


MessageBuilder = Callable[[AgentState], list[dict[str, str]]]
RouteFn = Callable[[AgentState], RouteDecision]


def default_message_builder(state: AgentState) -> list[dict[str, str]]:
    """Training-shaped few-shot prompt with no revision feedback (T2.3 adds that)."""
    return build_prompt(state["input"], state.get("exemplars") or [])


def default_route(state: AgentState) -> RouteDecision:
    """Pre-registered skeleton rules. Threshold comes from state, not a literal here."""
    if state.get("halt"):
        return "accept"
    iterations = int(state.get("iter") or 0)
    limit = int(state.get("max_generations") or MAX_GENERATIONS)
    candidates = state.get("candidates") or []
    if candidates and not candidates[-1].strip() and iterations < limit:
        return "revise"
    violations = state.get("last_violations") or []
    if violations and iterations < limit:
        return "revise"
    score = state.get("last_score")
    threshold = state.get("score_threshold")
    if (
        not violations
        and score is not None
        and threshold is not None
        and score < float(threshold)
        and iterations < limit
        and not state.get("re_retrieved", False)
    ):
        return "re_retrieve"
    return "accept"


def _append_trace(state: AgentState, node: str) -> list[TraceEntry]:
    entry = TraceEntry(node=node, round=int(state.get("iter") or 0))
    return [*list(state.get("trace") or []), entry]


def make_retrieve(retriever: Retriever) -> Callable[[AgentState], AgentState]:
    def retrieve(state: AgentState) -> AgentState:
        exemplars = list(retriever.retrieve(state["input"]))
        update: AgentState = {
            "exemplars": exemplars,
            "trace": _append_trace(state, "retrieve"),
        }
        if int(state.get("iter") or 0) > 0:
            update["re_retrieved"] = True
        return update

    return retrieve


def make_generate(
    generator: Generator,
    message_builder: MessageBuilder,
) -> Callable[[AgentState], AgentState]:
    def generate(state: AgentState) -> AgentState:
        trace = _append_trace(state, "generate")
        iterations = int(state.get("iter") or 0)
        try:
            messages = message_builder(state)
            text = generator.generate(messages)
        except Exception as exc:
            kind: FallbackKind = "first_round_error" if iterations == 0 else "later_round_error"
            error_type = type(exc).__name__
            LOGGER.error(
                "agent_generate_failed",
                round=iterations + 1,
                error_type=error_type,
            )
            return {
                "iter": iterations + 1,
                "halt": True,
                "halt_kind": kind,
                "error": f"{error_type}: {exc}",
                "trace": trace,
            }
        return {
            "candidates": [*list(state.get("candidates") or []), text],
            "iter": iterations + 1,
            "trace": trace,
        }

    return generate


def make_verify(verifier: Verifier) -> Callable[[AgentState], AgentState]:
    def verify(state: AgentState) -> AgentState:
        trace = _append_trace(state, "verify")
        if state.get("halt"):
            return {"trace": trace}
        candidates = state.get("candidates") or []
        if not candidates:
            return {"trace": trace}
        if not candidates[-1].strip():
            return {
                "last_violations": [],
                "violation_counts": [*list(state.get("violation_counts") or []), 0],
                "trace": trace,
            }
        violations = verifier.verify(state["input"], candidates[-1])
        return {
            "last_violations": violations,
            "violation_counts": [
                *list(state.get("violation_counts") or []),
                len(violations),
            ],
            "trace": trace,
        }

    return verify


def make_score(scorer: Scorer) -> Callable[[AgentState], AgentState]:
    def score_node(state: AgentState) -> AgentState:
        trace = _append_trace(state, "score")
        if state.get("halt"):
            return {"trace": trace}
        candidates = state.get("candidates") or []
        if not candidates:
            return {"trace": trace}
        if not candidates[-1].strip():
            return {
                "last_score": 0.0,
                "scores": [*list(state.get("scores") or []), 0.0],
                "trace": trace,
            }
        value = float(scorer.score(state["input"], candidates[-1]))
        return {
            "last_score": value,
            "scores": [*list(state.get("scores") or []), value],
            "trace": trace,
        }

    return score_node


def make_route_node() -> Callable[[AgentState], AgentState]:
    """Record the visit; the conditional edge chooses the branch separately."""

    def route_node(state: AgentState) -> AgentState:
        return {"trace": _append_trace(state, "route")}

    return route_node
