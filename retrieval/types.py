"""Shared retrieval records. Later routes return the same Hit shape for RRF."""

from typing import Protocol

import numpy as np
from pydantic import BaseModel


class Document(BaseModel):
    """One original text supplied by the caller."""

    id: str
    text: str


class Hit(BaseModel):
    """One ranked hit. `rank` starts at 1 after exclusions."""

    id: str
    rank: int
    score: float


class FusedHit(BaseModel):
    """One document after fusion. `sources` maps a route name to its rank in that route."""

    id: str
    rank: int
    score: float
    sources: dict[str, int]


class Embedder(Protocol):
    """Injected text encoder. `encode` returns shape `(n, dim)`."""

    def encode(self, texts: list[str]) -> np.ndarray: ...
