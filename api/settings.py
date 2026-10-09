"""Service settings read from the environment. No model or corpus is touched here."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from agent.config import AgentConfig

ENV_PREFIX = "SUTONG_"


class ServiceSettings(BaseModel):
    """Where the model server is and which local files the pipeline is assembled from."""

    # Unset means no real pipeline: the placeholders answer and /healthz reports 503.
    model_base_url: str | None = None
    model_name: str = "sutong-v2"
    pairs: Path = Path("corpus/pairs.jsonl")
    split: Path = Path("corpus/split.json")
    index_dir: Path = Path("retrieval/data")
    style_config: Path = Path("eval/configs/eval59.yaml")
    retrieval_config: Path = Path("eval/configs/retrieval.yaml")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServiceSettings:
        source = os.environ if environ is None else environ
        values = {
            name: source[ENV_PREFIX + name.upper()]
            for name in cls.model_fields
            if source.get(ENV_PREFIX + name.upper())
        }
        return cls.model_validate(values)


def service_agent_config() -> AgentConfig:
    """Loop settings the service runs with.

    Feedback goes into the system turn and copied-back feedback is cut before
    verification: of the formats compared in SPEC 4.9 this is the one that never
    returned the revision instructions to the caller.
    """
    return AgentConfig(feedback_format="system", echo_guard=True)
