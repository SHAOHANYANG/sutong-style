"""Agent loop knobs. Thresholds live here, not inside route branches."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.state import MAX_GENERATIONS


class AgentConfig(BaseModel):
    """Runtime settings. A null re-retrieve threshold disables that branch."""

    max_generations: int = Field(default=MAX_GENERATIONS, ge=1)
    re_retrieve_score_threshold: float | None = None
