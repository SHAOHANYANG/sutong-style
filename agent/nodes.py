"""Node callables and protocols. No I/O; callers inject every external capability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Protocol

import structlog

from agent.prompts import make_revision_message_builder, strip_feedback_echo
from agent.state import (
    MAX_GENERATIONS,
    AgentState,
    FallbackKind,
    RouteDecision,
)
from eval.fidelity import Violation
from retrieval.prompt import Exemplar

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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_messages(messages: list[dict[str, str]]) -> str:
    packed = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(packed)


def default_message_builder(state: AgentState) -> list[dict[str, str]]:
    """Revise when the previous round still has violations; otherwise plain few-shot."""
    raw = state.get("feedback_format") or "followup"
    if raw == "restate":
        return make_revision_message_builder("restate")(state)
    if raw == "followup":
        return make_revision_message_builder("followup")(state)
    if raw == "system":
        return make_revision_message_builder("system")(state)
    raise ValueError(f"未知反馈格式: {raw}")


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


def make_retrieve(retriever: Retriever) -> Callable[[AgentState], AgentState]:
    def retrieve(state: AgentState) -> AgentState:
        exemplars = list(retriever.retrieve(state["input"]))
        update: AgentState = {"exemplars": exemplars}
        if int(state.get("iter") or 0) > 0:
            update["re_retrieved"] = True
        return update

    return retrieve


def make_generate(
    generator: Generator,
    message_builder: MessageBuilder,
) -> Callable[[AgentState], AgentState]:
    def generate(state: AgentState) -> AgentState:
        iterations = int(state.get("iter") or 0)
        try:
            messages = message_builder(state)
            digest = sha256_messages(messages)
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
                "last_prompt_sha256": "",
                "last_output_sha256": "",
                "last_output_chars": 0,
                "last_generate_error_type": error_type,
                "last_echo_chars": 0,
            }
        removed = 0
        if state.get("echo_guard"):
            # The feedback this round answered was built from last_violations.
            text, removed = strip_feedback_echo(text, state.get("last_violations") or [])
        return {
            "candidates": [*list(state.get("candidates") or []), text],
            "iter": iterations + 1,
            "last_prompt_sha256": digest,
            "last_output_sha256": sha256_text(text),
            "last_output_chars": len(text),
            "last_generate_error_type": None,
            "last_echo_chars": removed,
            "echo_chars": [*list(state.get("echo_chars") or []), removed],
        }

    return generate


def make_verify(verifier: Verifier) -> Callable[[AgentState], AgentState]:
    def verify(state: AgentState) -> AgentState:
        if state.get("halt"):
            return {}
        candidates = state.get("candidates") or []
        if not candidates:
            return {}
        if not candidates[-1].strip():
            empty: list[Violation] = []
            return {
                "last_violations": empty,
                "violation_counts": [*list(state.get("violation_counts") or []), 0],
                "violations_history": [*list(state.get("violations_history") or []), empty],
            }
        violations = verifier.verify(state["input"], candidates[-1])
        return {
            "last_violations": violations,
            "violation_counts": [
                *list(state.get("violation_counts") or []),
                len(violations),
            ],
            "violations_history": [
                *list(state.get("violations_history") or []),
                list(violations),
            ],
        }

    return verify


def make_score(scorer: Scorer) -> Callable[[AgentState], AgentState]:
    def score_node(state: AgentState) -> AgentState:
        if state.get("halt"):
            return {}
        candidates = state.get("candidates") or []
        if not candidates:
            return {}
        if not candidates[-1].strip():
            return {
                "last_score": 0.0,
                "scores": [*list(state.get("scores") or []), 0.0],
            }
        value = float(scorer.score(state["input"], candidates[-1]))
        return {
            "last_score": value,
            "scores": [*list(state.get("scores") or []), value],
        }

    return score_node


def make_route_node(route_fn: RouteFn | None = None) -> Callable[[AgentState], AgentState]:
    """Record the decision; the conditional edge uses the same function."""

    choose = route_fn or default_route

    def route_node(state: AgentState) -> AgentState:
        return {"last_route": choose(state)}

    return route_node
