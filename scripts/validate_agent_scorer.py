"""Validate online scorer candidates against profile_gap_per_case on eval generations.

Numbers only. Same inputs must reproduce the same report aside from the timestamp.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from eval.metrics import profile_gap_per_case
from eval.run_eval import load_config, load_generations, load_style_reference, summarize
from retrieval.style_predictor import fit_predictor
from scripts.train import load_pairs, select
from stylometry.distance import StyleReference

LOGGER = structlog.get_logger()


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Rank correlation. Ties get average ranks. n < 2 or zero variance → 0.0."""
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.size != y.size:
        raise ValueError("Spearman 两侧长度不一致")
    if x.size < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = average
        i = j + 1
    return ranks


def candidate_a(reference: StyleReference, text: str) -> float:
    return abs(reference.distance(text) - 1.0)


def candidate_b(
    reference: StyleReference,
    predictor: Any,
    vernacular: str,
    text: str,
) -> float:
    target = predictor.predict(reference.zscore(vernacular))
    if getattr(target, "ndim", 1) == 2:
        target = target[0]
    return float(np.mean(np.abs(reference.zscore(text) - target)))


def pick_rate(
    truth: Mapping[str, Mapping[str, float]],
    scores: Mapping[str, Mapping[str, float]],
    *,
    lower_is_better: bool,
) -> float:
    """Among vernacular/base/lora, how often the scorer picks the best truth item."""
    labels = ("vernacular", "base", "lora")
    if not truth:
        return 0.0
    hits = 0
    for case_id, truth_row in truth.items():
        best_label = min(labels, key=lambda name: truth_row[name])
        scored = {name: scores[name][case_id] for name in labels}
        if lower_is_better:
            picked = min(labels, key=lambda name: scored[name])
        else:
            picked = max(labels, key=lambda name: scored[name])
        if picked == best_label:
            hits += 1
    return hits / len(truth)


def run_validation(
    *,
    pairs_path: Path,
    config_path: Path,
    base_path: Path,
    lora_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    reference = load_style_reference(config)
    pairs = load_pairs(pairs_path)
    train = select(pairs, "train")
    evaluate = select(pairs, "eval")
    predictor = fit_predictor(
        np.vstack([reference.zscore(row.vernacular) for row in train]),
        np.vstack([reference.zscore(row.original) for row in train]),
        [row.work for row in train],
    )
    base = load_generations(base_path)
    lora = load_generations(lora_path)

    truth: dict[str, dict[str, float]] = {}
    cand_a: dict[str, dict[str, float]] = {
        "vernacular": {},
        "base": {},
        "lora": {},
        "original": {},
    }
    cand_b: dict[str, dict[str, float]] = {
        "vernacular": {},
        "base": {},
        "lora": {},
        "original": {},
    }
    truth_lists: dict[str, list[float]] = {
        "vernacular": [],
        "base": [],
        "lora": [],
        "original": [],
    }
    a_lists: dict[str, list[float]] = {key: [] for key in cand_a}
    b_lists: dict[str, list[float]] = {key: [] for key in cand_b}

    for row in evaluate:
        if row.id not in base or row.id not in lora:
            raise SystemExit(f"缺少生成：{row.id}")
        texts = {
            "vernacular": row.vernacular,
            "base": base[row.id].output,
            "lora": lora[row.id].output,
            "original": row.original,
        }
        z_original = reference.zscore(row.original)
        case_truth: dict[str, float] = {}
        for name, text in texts.items():
            delta = np.asarray(reference.zscore(text) - z_original, dtype=np.float64).reshape(1, -1)
            gap = float(profile_gap_per_case(delta)[0])
            case_truth[name] = gap
            truth_lists[name].append(gap)
            a_val = candidate_a(reference, text)
            b_val = candidate_b(reference, predictor, row.vernacular, text)
            cand_a[name][row.id] = a_val
            cand_b[name][row.id] = b_val
            a_lists[name].append(a_val)
            b_lists[name].append(b_val)
        truth[row.id] = case_truth

    truth_for_pick = {
        case_id: {name: values[name] for name in ("vernacular", "base", "lora")}
        for case_id, values in truth.items()
    }
    report: dict[str, Any] = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "n_eval": len(evaluate),
        "n_train": len(train),
        "predictor_alpha": predictor.alpha,
        "profile_gap_per_case": {name: summarize(values) for name, values in truth_lists.items()},
        "candidate_a": {
            "values": {name: summarize(values) for name, values in a_lists.items()},
            "spearman_vs_truth": {
                name: spearman(truth_lists[name], a_lists[name])
                for name in ("vernacular", "base", "lora")
            },
            "pick_rate_among_three": pick_rate(
                truth_for_pick,
                {name: cand_a[name] for name in ("vernacular", "base", "lora")},
                lower_is_better=True,
            ),
        },
        "candidate_b": {
            "values": {name: summarize(values) for name, values in b_lists.items()},
            "spearman_vs_truth": {
                name: spearman(truth_lists[name], b_lists[name])
                for name in ("vernacular", "base", "lora")
            },
            "pick_rate_among_three": pick_rate(
                truth_for_pick,
                {name: cand_b[name] for name in ("vernacular", "base", "lora")},
                lower_is_better=True,
            ),
        },
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="校验 agent 在线打分候选")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("corpus/generations/base-eval59.jsonl"),
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=Path("corpus/generations/lora-eval59.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reports/agent-scorer-validation.json"),
    )
    args = parser.parse_args(None if argv is None else list(argv))
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    report = run_validation(
        pairs_path=args.pairs,
        config_path=args.config,
        base_path=args.base,
        lora_path=args.lora,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("agent_scorer_validation_written", path=str(args.output))


if __name__ == "__main__":
    main()
