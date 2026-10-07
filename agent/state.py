"""Graph state, public result model, and the two named safety limits."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from eval.fidelity import Violation
from retrieval.prompt import Exemplar

# Generator call cap. Route must accept once this many generations exist.
MAX_GENERATIONS = 3

# Longest legal path (node visits):
#   retrieve
#   + up to 3 * (generate -> verify -> score -> route)
#   + at most one extra retrieve on re_retrieve
# = 1 + 3 * 4 + 1 = 14.
# Margin of 6 catches a buggy router without waiting forever.
# RECURSION_LIMIT = 14 + 6 = 20.
RECURSION_LIMIT = 20

TerminationReason = Literal[
    "accepted",
    "max_rounds",
    "recursion_limit",
    "fallback",
]
FallbackKind = Literal[
    "first_round_error",
    "later_round_error",
    "all_blank",
]
RouteDecision = Literal["accept", "revise", "re_retrieve"]


class TraceEntry(BaseModel):
    """Minimal per-node record. Timestamps and payloads arrive in T2.4."""

    node: str
    round: int


class RoundRecord(BaseModel):
    """One generation attempt that produced text (possibly blank)."""

    round: int
    output: str
    violation_count: int
    score: float


class AgentResult(BaseModel):
    """Public return value. Failures stay visible; empty success is never claimed."""

    output: str
    selected_round: int | None
    total_rounds: int
    termination: TerminationReason
    fallback_kind: FallbackKind | None = None
    error: str | None = None
    rounds: list[RoundRecord] = Field(default_factory=list)
    re_retrieved: bool = False
    trace: list[TraceEntry] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """LangGraph state. Lists are replaced wholesale by each node return."""

    input: str
    exemplars: list[Exemplar]
    candidates: list[str]
    violation_counts: list[int]
    scores: list[float]
    last_violations: list[Violation]
    last_score: float
    iter: int
    re_retrieved: bool
    score_threshold: float
    trace: list[TraceEntry]
    halt: bool
    halt_kind: FallbackKind
    error: str
    hit_recursion_limit: bool
