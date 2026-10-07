"""Self-check rewrite loop. Nodes take injected protocols; no real models here."""

from agent.config import AgentConfig
from agent.graph import build_graph, run_agent
from agent.state import (
    MAX_GENERATIONS,
    RECURSION_LIMIT,
    AgentResult,
    RoundRecord,
    TraceEvent,
)

__all__ = [
    "MAX_GENERATIONS",
    "RECURSION_LIMIT",
    "AgentConfig",
    "AgentResult",
    "RoundRecord",
    "TraceEvent",
    "build_graph",
    "run_agent",
]
