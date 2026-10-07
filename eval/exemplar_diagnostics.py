"""Side effects of showing exemplars. Pure functions, no I/O.

exemplar_entity_leak is the part of a hallucination that also appears in the
exemplars. exemplar_copy_ratio is the closest exemplar original.
"""

from __future__ import annotations

from collections.abc import Sequence

from eval.fidelity import EntityTagger, extract_facts
from scripts.vernacularize import text_similarity


def exemplar_entity_leak(
    vernacular: str,
    output: str,
    exemplars: Sequence[tuple[str, str]],
    gazetteer: set[str],
    tagger: EntityTagger | None = None,
) -> float:
    """Fraction of output entities missing from the input but present in exemplars.

    Each exemplar is ``(vernacular, original)``. An output with no entities scores 0.
    """
    output_entities = extract_facts(output, gazetteer, tagger).entities
    if not output_entities:
        return 0.0
    source_entities = extract_facts(vernacular, gazetteer, tagger).entities
    seen: set[str] = set()
    for exemplar_vernacular, exemplar_original in exemplars:
        seen |= extract_facts(exemplar_vernacular, gazetteer, tagger).entities
        seen |= extract_facts(exemplar_original, gazetteer, tagger).entities
    leaked = (output_entities - source_entities) & seen
    return len(leaked) / len(output_entities)


def exemplar_copy_ratio(output: str, exemplar_originals: Sequence[str]) -> float:
    """Highest similarity between the output and any exemplar original. Empty is 0."""
    if not exemplar_originals:
        return 0.0
    return max(text_similarity(original, output) for original in exemplar_originals)
