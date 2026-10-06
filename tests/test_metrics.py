import numpy as np
import pytest

from eval.metrics import profile_gap, profile_gap_per_case
from stylometry.distance import StyleReference
from stylometry.features import FEATURE_NAMES
from stylometry.lexicon import LiteraryLexicon


def test_set_level_gap_cancels_opposite_errors() -> None:
    deltas = np.array([[1.0, -2.0], [-1.0, 2.0]], dtype=np.float64)
    per_case = profile_gap_per_case(deltas)
    assert profile_gap(deltas) == pytest.approx(0.0)
    assert per_case == pytest.approx([1.5, 1.5])
    assert float(per_case.mean()) > 0.0


def test_identical_profiles_are_zero() -> None:
    deltas = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    assert profile_gap(deltas) == pytest.approx(0.0)
    assert profile_gap_per_case(deltas) == pytest.approx([0.0, 0.0])


def test_same_text_reuses_zscore_and_both_gaps_are_zero() -> None:
    lexicon = LiteraryLexicon()
    text = "甲乙丙丁。戊己庚辛。"
    reference = StyleReference.fit([text, text], lexicon)
    scored = reference.zscore(text)
    assert reference.distance(text) == pytest.approx(
        float(np.linalg.norm(scored) / np.sqrt(len(FEATURE_NAMES)))
    )
    deltas = np.vstack([scored - scored])
    assert profile_gap(deltas) == pytest.approx(0.0)
    assert profile_gap_per_case(deltas) == pytest.approx([0.0])
