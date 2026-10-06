"""Pre-registered ablation of the style-vector predictor. Reads pairs, writes numbers."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog

from eval.metrics import profile_gap_per_case
from eval.run_eval import load_config, load_style_reference, summarize
from retrieval.style_index import StyleIndex
from retrieval.style_predictor import SEED, assign_folds, fit, fit_predictor, predict, select_alpha
from retrieval.types import Hit
from scripts.train import load_pairs, select
from stylometry.distance import StyleReference
from stylometry.features import FEATURE_NAMES

LOGGER = structlog.get_logger()
GROUPS = ("predicted", "raw", "mean", "oracle", "random")
KS = (1, 3)
BOOTSTRAP_DRAWS = 10_000


def bootstrap_interval(
    diffs: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED
) -> dict[str, float]:
    """95% percentile interval of the resampled mean. Seeded, so a rerun matches."""
    values = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    picked = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = picked.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def _gap(index: StyleIndex, hits: Sequence[Hit], target: np.ndarray) -> float:
    deltas = np.vstack([index.vector(hit.id) - target for hit in hits])
    return float(np.mean(profile_gap_per_case(deltas)))


def _random_gap(index: StyleIndex, target: np.ndarray, exclude: str) -> float:
    rows = [index.vector(doc_id) for doc_id in index.ids if doc_id != exclude]
    deltas = np.vstack(rows) - target
    return float(np.mean(profile_gap_per_case(deltas)))


def _check_oracle(means: dict[str, float]) -> None:
    oracle = means["oracle"]
    for name, value in means.items():
        if name != "oracle" and value < oracle:
            raise SystemExit(f"oracle check failed: {name} mean {value} < oracle {oracle}")


def _score_queries(
    index: StyleIndex,
    query_ids: Sequence[str],
    predicted: np.ndarray,
    raw: np.ndarray,
    target: np.ndarray,
    mean_vector: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    stored = {k: {group: np.empty(len(query_ids)) for group in GROUPS} for k in KS}
    widest = max(KS)
    for row, doc_id in enumerate(query_ids):
        exclude = (doc_id,)
        retrieved = {
            "predicted": index.search(predicted[row], widest, exclude_ids=exclude),
            "raw": index.search(raw[row], widest, exclude_ids=exclude),
            "mean": index.search(mean_vector, widest, exclude_ids=exclude),
            "oracle": index.search(target[row], widest, exclude_ids=exclude),
        }
        random_value = _random_gap(index, target[row], doc_id)
        for k in KS:
            for name, hits in retrieved.items():
                stored[k][name][row] = _gap(index, hits[:k], target[row])
            stored[k]["random"][row] = random_value
    return stored


def _pack(stored: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, dict[str, float | int]]]:
    packed: dict[str, dict[str, dict[str, float | int]]] = {}
    for k in KS:
        packed[f"k{k}"] = {group: summarize(stored[k][group].tolist()) for group in GROUPS}
    return packed


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    residual = ((y_true - y_pred) ** 2).sum(axis=0)
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    scores: dict[str, float | None] = {}
    for index, name in enumerate(FEATURE_NAMES):
        scores[name] = None if total[index] == 0.0 else float(1.0 - residual[index] / total[index])
    return scores


def _by_work(
    works: Sequence[str], stored: dict[int, dict[str, np.ndarray]]
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for work in sorted(set(works)):
        rows = [index for index, item in enumerate(works) if item == work]
        block: dict[str, object] = {"n": len(rows)}
        for k in KS:
            block[f"k{k}"] = {
                group: summarize([float(stored[k][group][index]) for index in rows])
                for group in GROUPS
            }
        block["predicted_minus_raw_mean"] = float(
            np.mean([stored[3]["predicted"][index] - stored[3]["raw"][index] for index in rows])
        )
        block["predicted_minus_mean_mean"] = float(
            np.mean([stored[3]["predicted"][index] - stored[3]["mean"][index] for index in rows])
        )
        grouped[work] = block
    return grouped


def run_ablation(
    *,
    train_ids: Sequence[str],
    train_works: Sequence[str],
    train_vernacular: np.ndarray,
    train_original: np.ndarray,
    eval_ids: Sequence[str],
    eval_works: Sequence[str],
    eval_vernacular: np.ndarray,
    eval_original: np.ndarray,
    commit: str,
    timestamp: str,
) -> dict[str, object]:
    """Score the five controls. Raises SystemExit when oracle is not the smallest."""
    if len(train_ids) != len(train_works):
        raise ValueError("train id 与 work 数量不一致")
    if not eval_ids:
        raise ValueError("eval 描述性结果缺少样本")
    folds = assign_folds(train_works, seed=SEED)
    index = StyleIndex(train_ids, train_original)
    oof = np.empty_like(train_original)
    stored = {k: {group: np.empty(len(train_ids)) for group in GROUPS} for k in KS}
    fold_alphas: list[dict[str, float | int]] = []
    for fold in range(5):
        valid = np.flatnonzero(folds == fold)
        if len(valid) == 0:
            continue
        train = np.flatnonzero(folds != fold)
        train_work_names = [train_works[int(index)] for index in train]
        alpha = select_alpha(
            train_vernacular[train],
            train_original[train],
            train_work_names,
            seed=SEED,
        )
        slope, intercept = fit(train_vernacular[train], train_original[train], alpha)
        oof[valid] = predict(train_vernacular[valid], slope, intercept)
        mean_vector = np.asarray(train_original[train].mean(axis=0), dtype=np.float64)
        fold_stored = _score_queries(
            index,
            [train_ids[int(index)] for index in valid],
            oof[valid],
            train_vernacular[valid],
            train_original[valid],
            mean_vector,
        )
        for k in KS:
            for group in GROUPS:
                stored[k][group][valid] = fold_stored[k][group]
        fold_alphas.append({"fold": fold, "n": len(valid), "alpha": alpha})

    packed = _pack(stored)
    for k in KS:
        _check_oracle({group: float(packed[f"k{k}"][group]["mean"]) for group in GROUPS})
    raw_gap = float(packed["k3"]["raw"]["mean"])
    mean_gap = float(packed["k3"]["mean"]["mean"])
    contrasted = {
        "predicted_minus_raw": bootstrap_interval(stored[3]["predicted"] - stored[3]["raw"]),
        "predicted_minus_mean": bootstrap_interval(stored[3]["predicted"] - stored[3]["mean"]),
    }
    advantage = contrasted["predicted_minus_raw"]["high"] < 0.0 and (
        contrasted["predicted_minus_mean"]["high"] < 0.0
    )
    full = fit_predictor(train_vernacular, train_original, train_works, seed=SEED)
    eval_predicted = full.predict(eval_vernacular)
    eval_index = StyleIndex(train_ids, train_original)
    eval_stored = _score_queries(
        eval_index,
        eval_ids,
        eval_predicted,
        eval_vernacular,
        eval_original,
        np.asarray(train_original.mean(axis=0), dtype=np.float64),
    )
    return {
        "advantage": advantage,
        "by_work": _by_work(train_works, stored),
        "commit": commit,
        "decision": "advantage" if advantage else "no_advantage",
        "eval": {
            "alpha": full.alpha,
            "k1": _pack(eval_stored)["k1"],
            "k3": _pack(eval_stored)["k3"],
            "n": len(eval_ids),
            "works": _by_work(eval_works, eval_stored),
        },
        "fold_alphas": fold_alphas,
        "paired_k3": contrasted,
        "parameters": {
            "alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "k_primary": 3,
            "k_reported": [1, 3],
            "n_eval": len(eval_ids),
            "n_folds": 5,
            "n_train": len(train_ids),
            "seed": SEED,
        },
        "r2": _r2(train_original, oof),
        "raw_worse_than_mean": raw_gap > mean_gap,
        "timestamp": timestamp,
        "train": packed,
    }


def _stack(reference: StyleReference, texts: Sequence[str]) -> np.ndarray:
    return np.vstack([reference.zscore(text) for text in texts])


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="风格向量预测的预注册消融")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reports/style-predictor-ablation.json"),
    )
    args = parser.parse_args(None if argv is None else list(argv))
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    config = load_config(args.config)
    reference = load_style_reference(config)
    rows = load_pairs(args.pairs)
    train_rows = select(rows, "train")
    eval_rows = select(rows, "eval")
    report = run_ablation(
        train_ids=[row.id for row in train_rows],
        train_works=[row.work for row in train_rows],
        train_vernacular=_stack(reference, [row.vernacular for row in train_rows]),
        train_original=_stack(reference, [row.original for row in train_rows]),
        eval_ids=[row.id for row in eval_rows],
        eval_works=[row.work for row in eval_rows],
        eval_vernacular=_stack(reference, [row.vernacular for row in eval_rows]),
        eval_original=_stack(reference, [row.original for row in eval_rows]),
        commit=_commit(),
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "style_ablation_written",
        n_train=len(train_rows),
        n_eval=len(eval_rows),
        advantage=bool(report["advantage"]),
        raw_worse_than_mean=bool(report["raw_worse_than_mean"]),
        output=str(args.output),
    )


if __name__ == "__main__":
    main()
