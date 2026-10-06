import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ablate_style_predictor import GROUPS, main, run_ablation
from stylometry.features import FEATURE_NAMES


def _basis(index: int) -> np.ndarray:
    vector = np.zeros(20, dtype=np.float64)
    vector[index] = 10.0
    return vector


def _run(timestamp: str) -> dict[str, object]:
    train = np.vstack([_basis(index) for index in range(6)])
    evaluation = np.vstack([_basis(index) for index in range(6, 8)])
    return run_ablation(
        train_ids=[f"t{index}" for index in range(6)],
        train_works=["甲"] * 6,
        train_vernacular=train.copy(),
        train_original=train,
        eval_ids=["e0", "e1"],
        eval_works=["乙", "乙"],
        eval_vernacular=evaluation.copy(),
        eval_original=evaluation,
        commit="abc",
        timestamp=timestamp,
    )


def test_bootstrap_interval_stays_negative_or_crosses_zero() -> None:
    from scripts.ablate_style_predictor import bootstrap_interval

    negative = bootstrap_interval(np.array([-2.0, -1.5, -1.0, -0.5]))
    assert negative["high"] < 0.0
    centered = bootstrap_interval(np.array([-1.0, -0.5, 0.5, 1.0]))
    assert centered["low"] < 0.0 < centered["high"]


def _mapping(report: dict[str, object], key: str) -> dict[str, object]:
    value = report[key]
    assert isinstance(value, dict)
    return value


def test_ablation_report_fields_and_repeat_without_the_timestamp() -> None:
    first = _run("2026-10-06T00:00:00Z")
    second = _run("2026-10-06T00:00:00Z")
    third = _run("2026-10-07T00:00:00Z")
    assert first == second
    first_body = {key: value for key, value in first.items() if key != "timestamp"}
    third_body = {key: value for key, value in third.items() if key != "timestamp"}
    assert first_body == third_body
    train = _mapping(first, "train")
    assert set(_mapping(train, "k3")) == set(GROUPS)
    assert set(_mapping(train, "k1")) == set(GROUPS)
    assert set(_mapping(first, "paired_k3")) == {"predicted_minus_raw", "predicted_minus_mean"}
    assert set(_mapping(first, "r2")) == set(FEATURE_NAMES)
    alphas = first["fold_alphas"]
    assert isinstance(alphas, list)
    assert len(alphas) == 5
    parameters = _mapping(first, "parameters")
    assert parameters["bootstrap_draws"] == 10_000
    assert parameters["seed"] == 42
    assert _mapping(first, "eval")["n"] == 2


def test_oracle_violation_exits() -> None:
    original = np.vstack(
        [
            np.zeros(20),
            np.array([5.0, 0.0, *np.zeros(18)]),
            np.array([3.0, 3.0, *np.zeros(18)]),
        ]
    )
    vernacular = np.vstack([original[1], original[0], original[0]])
    with pytest.raises(SystemExit, match="oracle"):
        run_ablation(
            train_ids=["a", "b", "c"],
            train_works=["甲", "甲", "甲"],
            train_vernacular=vernacular,
            train_original=original,
            eval_ids=["e"],
            eval_works=["乙"],
            eval_vernacular=np.vstack([np.zeros(20)]),
            eval_original=np.vstack([np.ones(20)]),
            commit="abc",
            timestamp="2026-10-06T00:00:00Z",
        )


def test_cli_writes_a_report_from_handwritten_pairs(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    rows = []
    for index in range(6):
        rows.append(
            {
                "id": f"t{index}",
                "work": "自编",
                "idx": index,
                "vernacular": f"第{index}天，张三把两袋米放在门口。",
                "original": f"第{index}日，张三置米两袋于门侧，雨却未停。",
                "split": "train",
            }
        )
    for index in range(2):
        rows.append(
            {
                "id": f"e{index}",
                "work": "自编",
                "idx": index,
                "vernacular": f"井水浑了{index}天，他去请人来修。",
                "original": f"井水浑浊已{index}日，他出门去请修井的人。",
                "split": "eval",
            }
        )
    pairs.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "out" / "report.json"
    main(
        [
            "--pairs",
            str(pairs),
            "--config",
            "eval/configs/eval59.yaml",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["train"]["k3"]) == set(GROUPS)
    assert payload["decision"] in {"advantage", "no_advantage"}
    assert "original" not in json.dumps(payload["train"])
