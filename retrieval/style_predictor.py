"""Closed-form ridge from vernacular z to original z. No scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stylometry.features import FEATURE_NAMES

DIM = len(FEATURE_NAMES)
ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
N_FOLDS = 5
SEED = 42


def assign_folds(works: Sequence[str], *, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Deal each work round-robin after a seeded shuffle. Works are independent."""
    folds = np.empty(len(works), dtype=np.int64)
    by_work: dict[str, list[int]] = {}
    for index, work in enumerate(works):
        by_work.setdefault(work, []).append(index)
    for work in sorted(by_work):
        order = np.array(by_work[work], dtype=np.int64)
        np.random.default_rng(seed).shuffle(order)
        for offset, index in enumerate(order):
            folds[int(index)] = offset % n_folds
    return folds


def fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope `(dim, dim)` and intercept `(dim,)`. The intercept is not penalized."""
    design = np.concatenate([np.asarray(x, dtype=np.float64), np.ones((len(x), 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    solved = np.linalg.solve(gram, design.T @ np.asarray(y, dtype=np.float64))
    coef: np.ndarray = np.asarray(solved, dtype=np.float64)
    slope_hat: np.ndarray = np.asarray(coef[:-1], dtype=np.float64)
    intercept_hat: np.ndarray = np.asarray(coef[-1], dtype=np.float64)
    return slope_hat, intercept_hat


def predict(x: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    values: np.ndarray = np.asarray(
        np.asarray(x, dtype=np.float64) @ slope + intercept, dtype=np.float64
    )
    return values


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    works: Sequence[str],
    *,
    alphas: Sequence[float] = ALPHAS,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> float:
    """Smallest alpha among those with the best pooled out-of-fold mean squared error."""
    folds = assign_folds(works, n_folds=n_folds, seed=seed)
    best_alpha = float(alphas[0])
    best_mse = float("inf")
    for alpha in alphas:
        squared: list[np.ndarray] = []
        for fold in range(n_folds):
            valid = folds == fold
            if not np.any(valid):
                continue
            train = ~valid
            if not np.any(train):
                raise ValueError("训练折为空")
            slope, intercept = fit(x[train], y[train], float(alpha))
            residual = predict(x[valid], slope, intercept) - y[valid]
            squared.append(residual.ravel() ** 2)
        mse = float(np.mean(np.concatenate(squared)))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


class StylePredictor:
    """A fitted map. `predict` accepts one vector or a matrix."""

    def __init__(self, slope: np.ndarray, intercept: np.ndarray, alpha: float) -> None:
        self.slope = np.asarray(slope, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.alpha = float(alpha)

    def predict(self, x: np.ndarray) -> np.ndarray:
        point = np.asarray(x, dtype=np.float64)
        if point.ndim == 1:
            row = predict(point.reshape(1, -1), self.slope, self.intercept)[0]
            return np.asarray(row, dtype=np.float64)
        return np.asarray(predict(point, self.slope, self.intercept), dtype=np.float64)


def fit_predictor(
    x: np.ndarray,
    y: np.ndarray,
    works: Sequence[str],
    *,
    seed: int = SEED,
) -> StylePredictor:
    alpha = select_alpha(x, y, works, seed=seed)
    slope, intercept = fit(x, y, alpha)
    return StylePredictor(slope, intercept, alpha)
