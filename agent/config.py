"""Agent loop knobs. Thresholds and feedback format live here, not in branches."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.prompts import FeedbackFormat
from agent.state import MAX_GENERATIONS


class AgentConfig(BaseModel):
    """Runtime settings. A null re-retrieve threshold disables that branch."""

    max_generations: int = Field(default=MAX_GENERATIONS, ge=1)
    re_retrieve_score_threshold: float | None = None
    feedback_format: FeedbackFormat = "followup"
