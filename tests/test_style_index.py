import numpy as np
import pytest

from retrieval.style_index import StyleIndex


def _row(*values: float) -> np.ndarray:
    vector = np.zeros(20, dtype=np.float64)
    vector[: len(values)] = values
    return vector


def test_nearest_neighbors_break_ties_by_id() -> None:
    index = StyleIndex(
        ["b", "a", "c"],
        np.vstack([_row(0.0, 1.0), _row(0.0, 0.0), _row(1.0, 0.0)]),
    )
    hits = index.search(_row(0.0, 0.0), k=3)
    assert [(hit.id, hit.rank) for hit in hits] == [("a", 1), ("b", 2), ("c", 3)]
    assert [hit.score for hit in hits] == pytest.approx([0.0, -1.0, -1.0])


def test_exclude_ids_renumbers_ranks_from_one() -> None:
    index = StyleIndex(["a", "b"], np.vstack([_row(0.0), _row(1.0)]))
    hits = index.search(_row(0.0), k=2, exclude_ids=["a"])
    assert [(hit.id, hit.rank) for hit in hits] == [("b", 1)]


def test_k_boundaries_and_empty_index() -> None:
    index = StyleIndex(["a"], np.vstack([_row(1.0)]))
    assert len(index.search(_row(0.0), k=5)) == 1
    with pytest.raises(ValueError, match="正整数"):
        index.search(_row(0.0), k=0)
    empty = StyleIndex([], np.zeros((0, 20)))
    assert empty.search(_row(0.0), k=3) == []


def test_duplicate_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="重复"):
        StyleIndex(["a", "a"], np.vstack([_row(0.0), _row(1.0)]))


def test_row_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="行数"):
        StyleIndex(["a", "b"], np.vstack([_row(0.0)]))


def test_non_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="二维"):
        StyleIndex(["a"], _row(0.0))


def test_non_finite_values_are_rejected() -> None:
    row = _row(0.0)
    row[0] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        StyleIndex(["a"], np.vstack([row]))
