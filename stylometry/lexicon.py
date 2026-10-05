"""Literary and colloquial word scores learned from parallel counts."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pydantic import BaseModel, Field


class LiteraryLexicon(BaseModel):
    """Word-to-score maps. Scores only; no source sentences."""

    literary: dict[str, float] = Field(default_factory=dict)
    colloquial: dict[str, float] = Field(default_factory=dict)

    def literary_hit_rate(self, tokens: list[str]) -> float:
        """Share of tokens that appear in the literary list."""
        if not tokens:
            return 0.0
        hits = sum(1 for token in tokens if token in self.literary)
        return hits / len(tokens)


def build_lexicon(
    original_counts: Mapping[str, int],
    vernacular_counts: Mapping[str, int],
) -> LiteraryLexicon:
    """Keep literary words with score > 1 and original count >= 3.

    Colloquial words use the symmetric rule: score < -1 and vernacular count >= 3.
    ``score = log((count_original + 1) / (count_vernacular + 1))``.
    """
    literary: dict[str, float] = {}
    colloquial: dict[str, float] = {}
    for word in set(original_counts) | set(vernacular_counts):
        original = original_counts.get(word, 0)
        vernacular = vernacular_counts.get(word, 0)
        score = math.log((original + 1) / (vernacular + 1))
        if score > 1.0 and original >= 3:
            literary[word] = score
        elif score < -1.0 and vernacular >= 3:
            colloquial[word] = score
    return LiteraryLexicon(literary=literary, colloquial=colloquial)


def lexicon_payload(lexicon: LiteraryLexicon) -> dict[str, dict[str, float]]:
    """Stable JSON body: sorted words and scores, nothing else."""
    return {
        "literary": dict(sorted(lexicon.literary.items())),
        "colloquial": dict(sorted(lexicon.colloquial.items())),
    }
