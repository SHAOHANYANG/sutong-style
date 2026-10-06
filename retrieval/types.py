"""Shared retrieval records. Later routes return the same Hit shape for RRF."""

from pydantic import BaseModel


class Document(BaseModel):
    """One original text supplied by the caller."""

    id: str
    text: str


class Hit(BaseModel):
    """One positive-score hit. `rank` starts at 1 after exclusions."""

    id: str
    rank: int
    score: float
