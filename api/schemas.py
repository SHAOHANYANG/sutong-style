"""Pydantic request, response, and SSE payload schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from agent.state import AgentResult

MAX_TEXT_CHARS = 10_000


class TransformRequest(BaseModel):
    """One style-transfer request."""

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    style: str = "sutong"
    stream: bool = True
    max_iter: int = Field(default=3, ge=1, le=3)
    seed: int = 42

    @field_validator("text")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text 不能只包含空白字符")
        return value


class TransformResponse(AgentResult):
    """AgentResult plus the request seed recorded at the service boundary."""

    seed: int


class TokenEvent(BaseModel):
    """A fixed-size piece of one generation round."""

    round: int
    text: str


class ErrorEvent(BaseModel):
    """Terminal SSE error that does not expose request prose."""

    error_type: str
    detail: str


class DependencyStatus(BaseModel):
    """Readiness of one injected service."""

    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate dependency readiness."""

    status: str
    dependencies: dict[str, DependencyStatus]
