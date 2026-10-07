"""Fidelity verifier. Gazetteer and tagger are injected; this module does no I/O."""

from __future__ import annotations

from eval.fidelity import EntityTagger, Violation, assess


class FidelityVerifier:
    """Adapter from assess() onto the agent Verifier protocol."""

    def __init__(
        self,
        gazetteer: set[str],
        *,
        tagger: EntityTagger | None = None,
    ) -> None:
        self.gazetteer = gazetteer
        self.tagger = tagger

    def verify(self, vernacular: str, output: str) -> list[Violation]:
        return assess(vernacular, output, self.gazetteer, self.tagger).violations
