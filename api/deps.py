"""FastAPI dependency assembly points.

T3.1 deliberately ships no real model client. T3.7 replaces these providers with
vLLM/OpenAI-compatible implementations; tests replace them through
``app.dependency_overrides``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.nodes import Generator, Retriever, Scorer, Verifier
from eval.fidelity import Violation
from retrieval.prompt import Exemplar


class StyleConfig(BaseModel):
    """Adapters exposed by this API process."""

    adapters: list[str] = Field(default_factory=lambda: ["sutong"])


class _UnavailableRetriever:
    def retrieve(self, vernacular: str) -> list[Exemplar]:
        raise NotImplementedError("real retriever assembly is implemented in a later task")

    def ping(self) -> bool:
        return False


class _UnavailableGenerator:
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError("real vLLM generator assembly is implemented in T3.7")

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


def get_retriever() -> Retriever:
    """Return the configured retriever; override this dependency in tests."""
    return _RETRIEVER


def get_generator() -> Generator:
    """Return the configured generator; T3.7 replaces the placeholder."""
    return _GENERATOR


def get_verifier() -> Verifier:
    """Return the configured deterministic verifier."""
    return _VERIFIER


def get_scorer() -> Scorer:
    """Return the configured deterministic scorer."""
    return _SCORER


def get_style_config() -> StyleConfig:
    """Return the configured adapter names."""
    return _STYLES
