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


class Embedder(Protocol):
    """Injected text encoder. `encode` returns shape `(n, dim)`."""

    def encode(self, texts: list[str]) -> np.ndarray: ...
