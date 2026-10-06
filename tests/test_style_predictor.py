import numpy as np
import pytest

from retrieval.style_predictor import assign_folds, fit, predict, select_alpha


def test_small_alpha_recovers_a_known_map() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 20))
    slope = rng.normal(size=(20, 20))
    intercept = rng.normal(size=20)
    y = x @ slope + intercept
    learned_slope, learned_intercept = fit(x, y, alpha=1e-8)
    assert learned_slope == pytest.approx(slope, abs=1e-4)
    assert learned_intercept == pytest.approx(intercept, abs=1e-4)


def test_large_alpha_shrinks_predictions_to_the_target_mean() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 20))
    y = rng.normal(size=(40, 20)) + 4.0
    slope, intercept = fit(x, y, alpha=1e12)
    learned = predict(x, slope, intercept)
    assert learned == pytest.approx(np.broadcast_to(y.mean(axis=0), y.shape), abs=1e-3)
    assert learned.mean() == pytest.approx(4.0, abs=0.2)
    assert not np.allclose(learned.mean(), 0.0)


def test_select_alpha_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 20))
    y = x @ rng.normal(size=(20, 20)) + rng.normal(size=20)
    works = ["甲"] * 15 + ["乙"] * 15
    assert select_alpha(x, y, works, seed=42) == select_alpha(x, y, works, seed=42)


def test_folds_cover_every_work_and_repeat() -> None:
    works = ["乙"] * 5 + ["甲"] * 5
    first = assign_folds(works, seed=42)
    second = assign_folds(works, seed=42)
    assert np.array_equal(first, second)
    for start in (0, 5):
        assert set(first[start : start + 5].tolist()) == {0, 1, 2, 3, 4}


def test_select_alpha_rejects_an_empty_training_fold() -> None:
    with pytest.raises(ValueError, match="训练折为空"):
        select_alpha(np.zeros((1, 20)), np.zeros((1, 20)), ["甲"])
