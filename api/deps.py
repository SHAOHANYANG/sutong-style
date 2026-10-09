"""FastAPI dependency assembly points.

With SUTONG_MODEL_BASE_URL set, the providers return the real pipeline built by
api.assembly against a model server. Without it they return placeholders, so the
process still starts and /healthz reports 503. Tests replace the providers through
``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent.config import AgentConfig
from agent.nodes import Generator, Retriever, Scorer, Verifier
from api.settings import ServiceSettings, service_agent_config
from eval.fidelity import Violation
from retrieval.prompt import Exemplar

if TYPE_CHECKING:
    from api.assembly import Services


class StyleConfig(BaseModel):
    """Adapters exposed by this API process."""

    adapters: list[str] = Field(default_factory=lambda: ["sutong"])


class _UnavailableRetriever:
    def retrieve(self, vernacular: str) -> list[Exemplar]:
        raise NotImplementedError("no model server configured: set SUTONG_MODEL_BASE_URL")

    def ping(self) -> bool:
        return False


class _UnavailableGenerator:
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError("no model server configured: set SUTONG_MODEL_BASE_URL")

    def ping(self) -> bool:
        return False


class _UnavailableVerifier:
    def verify(self, vernacular: str, output: str) -> list[Violation]:
        raise NotImplementedError("real verifier assembly requires project data")


class _UnavailableScorer:
    def score(self, vernacular: str, output: str) -> float:
        raise NotImplementedError("real scorer assembly requires project data")


_RETRIEVER = _UnavailableRetriever()
_GENERATOR = _UnavailableGenerator()
_VERIFIER = _UnavailableVerifier()
_SCORER = _UnavailableScorer()
_STYLES = StyleConfig()


@lru_cache(maxsize=1)
def services() -> Services | None:
    """The real pipeline, built once. None when no model server is configured."""
    settings = ServiceSettings.from_env()
    if settings.model_base_url is None:
        return None
    # Imported here: assembly reads the corpus and the indexes.
    from api.assembly import build_services

    return build_services(settings)


def get_retriever() -> Retriever:
    """Return the configured retriever; override this dependency in tests."""
    built = services()
    return _RETRIEVER if built is None else built.retriever


def get_generator() -> Generator:
    """Return the configured generator; override this dependency in tests."""
    built = services()
    return _GENERATOR if built is None else built.generator


def get_verifier() -> Verifier:
    """Return the configured deterministic verifier."""
    built = services()
    return _VERIFIER if built is None else built.verifier


def get_scorer() -> Scorer:
    """Return the configured deterministic scorer."""
    built = services()
    return _SCORER if built is None else built.scorer


def get_agent_config() -> AgentConfig:
    """Loop settings for a request, before max_iter is applied."""
    return service_agent_config()


def get_style_config() -> StyleConfig:
    """Return the configured adapter names."""
    return _STYLES
