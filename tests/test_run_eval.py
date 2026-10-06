import json
import socket
from pathlib import Path

import pytest

from eval.run_eval import (
    AGGREGATE_FIELDS,
    EvalConfig,
    build_judge_completer,
    evaluate,
    load_config,
    main,
)
from stylometry.features import FEATURE_NAMES
from tests.fakes import FakeGenerator
from tests.test_judge import AlwaysA


def _generations(path: Path, *, suffix: str = "") -> None:
    cases = [
        json.loads(line)
        for line in Path("corpus/sample_public.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fake = FakeGenerator()
    lines = []
    for case in cases:
        output = fake.generate(f"原文：\n{case['vernacular']}", seed=42) + suffix
        lines.append(
            json.dumps(
                {
                    "id": case["id"],
                    "model": "fake",
                    "pipeline": "baseline",
                    "output": output,
                    "trace": None,
                    "seed": 42,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert fake.calls


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("outbound network")


def test_skip_judge_makes_no_network_request_and_writes_six_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("judge client constructed")

    monkeypatch.setattr("eval.run_eval.build_judge_completer", boom)
    generations = tmp_path / "generations.jsonl"
    report_path = tmp_path / "report.json"
    _generations(generations)
    main(
        [
            "--config",
            "eval/configs/baseline.yaml",
            "--generations",
            str(generations),
            "--run-id",
            "smoke-skip",
            "--skip-judge",
            "--report",
            str(report_path),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert list(payload["aggregate"]) == list(AGGREGATE_FIELDS)
    assert payload["n_cases"] == 5
    assert len(payload["per_case"]) == 5
    assert payload["aggregate"]["style_win_rate"] is None
    assert payload["distribution"]["style_win_rate"] is None
    assert payload["aggregate"]["ppl"] is None
    assert payload["aggregate"]["copy_ratio"] == 1.0
    assert payload["model"] == "fake"
    for name in (
        "style_distance",
        "profile_gap_per_case",
        "entity_recall",
        "numeral_recall",
        "hallucination_rate",
    ):
        assert isinstance(payload["aggregate"][name], float)
        assert set(payload["distribution"][name]) == {
            "mean",
            "std",
            "median",
            "q1",
            "q3",
            "min",
            "max",
        }
    assert set(payload["distribution"]["copy_ratio"]) == {
        "mean",
        "std",
        "median",
        "q1",
        "q3",
        "min",
        "max",
    }
    assert payload["distribution"]["copy_ratio"]["mean"] == 1.0
    assert payload["distribution"]["profile_gap"] is None
    assert isinstance(payload["aggregate"]["profile_gap"], float)
    assert isinstance(payload["aggregate"]["profile_gap_per_case"], float)
    assert list(payload["profile_mean_delta"]) == list(FEATURE_NAMES)
    assert list(payload["vernacular_input"]["profile_mean_delta"]) == list(FEATURE_NAMES)
    assert isinstance(payload["vernacular_input"]["profile_gap"], float)
    assert isinstance(payload["vernacular_input"]["profile_gap_per_case"], float)
    for case in payload["per_case"]:
        assert case["output_hash"].startswith("sha256:")
        assert isinstance(case["metrics"]["profile_gap_per_case"], float)
        assert case["metrics"]["copy_ratio"] == 1.0
        assert "output" not in case
        assert "original" not in case
        assert "vernacular" not in case
    assert not (tmp_path / "cache").exists()


def test_identical_output_skips_the_judge(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    _generations(generations)
    judge = AlwaysA()
    report = evaluate(
        load_config(Path("eval/configs/baseline.yaml")),
        generations,
        "identical",
        skip_judge=False,
        report_path=tmp_path / "report.json",
        cache_dir=tmp_path / "cache",
        completer=judge,
    )
    tally = report.distribution["style_win_rate"]
    assert judge.calls == 0
    assert tally is not None
    assert tally["identical_rate"] == 1.0
    assert tally["tie"] == 0
    assert tally["n"] == 5
    assert report.aggregate["style_win_rate"] == 0.5
    copied = report.distribution["copy_ratio"]
    assert copied is not None
    assert copied["mean"] == 1.0


def test_distinct_output_asks_twice_per_case(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    _generations(generations, suffix="另写了一句。")
    judge = AlwaysA()
    report = evaluate(
        load_config(Path("eval/configs/baseline.yaml")),
        generations,
        "distinct",
        skip_judge=False,
        report_path=tmp_path / "report.json",
        cache_dir=tmp_path / "cache",
        completer=judge,
    )
    tally = report.distribution["style_win_rate"]
    assert tally is not None
    assert tally["identical_rate"] == 0.0
    assert judge.calls == 2 * tally["n"]


def test_missing_generation_is_reported(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        json.dumps({"id": "missing", "output": "无"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample_001"):
        evaluate(
            load_config(Path("eval/configs/baseline.yaml")),
            generations,
            "incomplete",
            skip_judge=True,
            report_path=tmp_path / "report.json",
            cache_dir=tmp_path / "cache",
        )


def test_judge_client_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    created: dict[str, str | None] = {}

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
            created["api_key"] = api_key
            created["base_url"] = base_url
            created["model"] = model

        def generate(self, prompt: str, **kwargs: object) -> str:
            return "A"

    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr("infra.openai_generator.OpenAICompatibleGenerator", FakeClient)
    completer = build_judge_completer("deepseek-v4-pro")
    assert completer.generate("提示") == "A"
    assert created == {
        "api_key": "test-key",
        "base_url": "http://127.0.0.1:9/v1",
        "model": "deepseek-v4-pro",
    }


def test_judge_client_refuses_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        build_judge_completer("deepseek-v4-pro")


def test_split_filter_keeps_only_matching_cases(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "id": "a",
                    "vernacular": "甲去了苏州，带了3本书。",
                    "original": "甲赴苏州，携书三。",
                    "split": "train",
                },
                {
                    "id": "b",
                    "vernacular": "乙在南京住了5年。",
                    "original": "乙居南京五载。",
                    "split": "eval",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_config(Path("eval/configs/baseline.yaml")).model_copy(
        update={"cases": str(cases), "split": "eval"}
    )
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        json.dumps({"id": "b", "model": "fake", "output": "乙居南京五载。"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    report = evaluate(
        config,
        generations,
        "split-filter",
        skip_judge=True,
        report_path=tmp_path / "report.json",
        cache_dir=tmp_path / "cache",
    )
    assert report.n_cases == 1
    assert [case.id for case in report.per_case] == ["b"]
    assert report.aggregate["profile_gap"] == pytest.approx(0.0)
    assert report.aggregate["profile_gap_per_case"] == pytest.approx(0.0)
    assert report.per_case[0].metrics["profile_gap_per_case"] == pytest.approx(0.0)


def _hand_config(tmp_path: Path, rows: list[dict[str, str]]) -> EvalConfig:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return load_config(Path("eval/configs/baseline.yaml")).model_copy(update={"cases": str(cases)})


def test_profile_fields_are_null_without_ground_truth(tmp_path: Path) -> None:
    config = _hand_config(
        tmp_path,
        [{"id": "h1", "vernacular": "今天下雨。", "original": ""}],
    )
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        json.dumps({"id": "h1", "output": "雨还在下。"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = evaluate(
        config,
        generations,
        "no-original",
        skip_judge=True,
        report_path=tmp_path / "report.json",
        cache_dir=tmp_path / "cache",
    )
    assert report.aggregate["profile_gap"] is None
    assert report.aggregate["profile_gap_per_case"] is None
    assert report.distribution["profile_gap"] is None
    assert report.distribution["profile_gap_per_case"] is None
    assert report.profile_mean_delta is None
    assert report.vernacular_input.profile_gap is None
    assert report.vernacular_input.profile_gap_per_case is None
    assert report.vernacular_input.profile_mean_delta is None
    assert report.per_case[0].metrics["profile_gap_per_case"] is None
    assert isinstance(report.aggregate["style_distance"], float)


def test_mixed_originals_are_rejected(tmp_path: Path) -> None:
    config = _hand_config(
        tmp_path,
        [
            {"id": "a", "vernacular": "今天下雨。", "original": "雨落了一日。"},
            {"id": "b", "vernacular": "他去了码头。", "original": ""},
        ],
    )
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"id": "a", "output": "雨落了一日。"},
                {"id": "b", "output": "他去了码头。"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="一部分有原文"):
        evaluate(
            config,
            generations,
            "mixed",
            skip_judge=True,
            report_path=tmp_path / "report.json",
            cache_dir=tmp_path / "cache",
        )


def test_unknown_split_is_an_error(tmp_path: Path) -> None:
    config = load_config(Path("eval/configs/baseline.yaml")).model_copy(update={"split": "nope"})
    generations = tmp_path / "generations.jsonl"
    generations.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="nope"):
        evaluate(
            config,
            generations,
            "bad-split",
            skip_judge=True,
            report_path=tmp_path / "report.json",
            cache_dir=tmp_path / "cache",
        )
