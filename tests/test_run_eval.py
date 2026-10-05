import json
import socket
from pathlib import Path

import pytest

from eval.run_eval import build_judge_completer, evaluate, load_config, main
from tests.fakes import FakeGenerator
from tests.test_judge import AlwaysA


def _generations(path: Path) -> None:
    cases = [
        json.loads(line)
        for line in Path("corpus/sample_public.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fake = FakeGenerator()
    lines = []
    for case in cases:
        output = fake.generate(f"原文：\n{case['vernacular']}", seed=42)
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
    assert set(payload["aggregate"]) == {
        "style_distance",
        "entity_recall",
        "numeral_recall",
        "hallucination_rate",
        "style_win_rate",
        "ppl",
    }
    assert payload["n_cases"] == 5
    assert len(payload["per_case"]) == 5
    assert payload["aggregate"]["style_win_rate"] is None
    assert payload["distribution"]["style_win_rate"] is None
    assert payload["aggregate"]["ppl"] is None
    assert payload["model"] == "fake"
    for name in ("style_distance", "entity_recall", "numeral_recall", "hallucination_rate"):
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
    for case in payload["per_case"]:
        assert case["output_hash"].startswith("sha256:")
        assert "output" not in case
        assert "original" not in case
        assert "vernacular" not in case
    assert not (tmp_path / "cache").exists()


def test_report_win_rate_stays_at_one_half_when_the_judge_always_picks_a(tmp_path: Path) -> None:
    generations = tmp_path / "generations.jsonl"
    _generations(generations)
    judge = AlwaysA()
    report = evaluate(
        load_config(Path("eval/configs/baseline.yaml")),
        generations,
        "always-a",
        skip_judge=False,
        report_path=tmp_path / "report.json",
        cache_dir=tmp_path / "cache",
        completer=judge,
    )
    tally = report.distribution["style_win_rate"]
    assert tally is not None
    assert tally["tie_rate"] == 1.0
    assert tally["win"] == 0
    assert tally["loss"] == 0
    assert tally["tie"] == 5
    assert tally["n"] == 5
    assert isinstance(tally["tie"], int)
    assert report.aggregate["style_win_rate"] == 0.5


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
