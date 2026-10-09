import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.prompts import FOLLOWUP_LEAD, RESTATE_LEAD, build_revision_messages
from eval.fidelity import Violation
from eval.run_eval import evaluate, load_config
from retrieval.prompt import Exemplar, build_prompt
from scripts.eval_agent import compare, exit_condition, feedback_echo
from scripts.eval_agent import main as eval_main
from scripts.run_agent_eval import (
    PRIMARY_ARM,
    arms,
    first_round_violation_cases,
    main,
    round_distribution,
    violation_transitions,
)
from scripts.sweep_topk import main as sweep_main
from scripts.train import Pair
from tests.fakes import FakeAgentScorer

ROOT = Path(__file__).resolve().parents[1]
# Made-up sentences. The marker must never reach a committed report.
MARKER = "独特标记丙丁"
DRIFTING = f"戊带着12块钱出门，{MARKER}"
CLEAN = "己关上了门"
FIRST_DRIFTING = "戊出门了"
FIRST_CLEAN = "门被己关上了"
REPAIRED = "戊带着十二块钱出门"


class _SweepGenerator:
    """Round-one outputs, written into the sweep files the agent run reads back."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        return FIRST_DRIFTING if messages[-1]["content"] == DRIFTING else FIRST_CLEAN


class _RevisionGenerator:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs or [REPAIRED]
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


class _FixedCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return 12


class _HugeCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return 10_000


def _pair(doc_id: str, split: str, vernacular: str, original: str) -> Pair:
    return Pair(
        id=doc_id, work="自编", idx=1, vernacular=vernacular, original=original, split=split
    )


def _world(tmp_path: Path) -> dict[str, Path]:
    pairs = [
        _pair("t1", "train", "甲去井边", "井边有一个人"),
        _pair("t2", "train", "乙买了米", "米放在门口"),
        _pair("t3", "train", "丙看着天", "天色暗下来"),
        _pair("e1", "eval", DRIFTING, "戊揣着十二块钱出了门"),
        _pair("e2", "eval", CLEAN, "门关上了"),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(row.model_dump_json() + "\n" for row in pairs), encoding="utf-8")
    plan = {
        "plans": {
            config: [
                {
                    "exemplars": [
                        {"id": item, "rank": rank, "score": 0.1, "sources": {"bm25": rank}}
                        for rank, item in enumerate(["t1", "t2", "t3"], start=1)
                    ],
                    "id": query,
                    "work": "自编",
                }
                for query in ("e1", "e2")
            ]
            for config in ("equal", "balanced", "content", "style")
        }
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    lora = tmp_path / "lora.jsonl"
    lora.write_text(
        json.dumps({"id": "e1", "output": FIRST_DRIFTING}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "e2", "output": FIRST_CLEAN}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    generations = tmp_path / "generations"
    sweep_main(
        [
            *("--plan", str(plan_path)),
            *("--pairs", str(pairs_path)),
            *("--baseline", str(lora)),
            *("--adapter", str(tmp_path / "sutong-v2" / "adapter")),
            *("--output-dir", str(generations)),
            *("--manifest", str(tmp_path / "sweep-manifest.json")),
        ],
        generator=_SweepGenerator(),
        counter=_FixedCounter(),
    )
    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                "pipeline: baseline",
                "model: sutong-v2",
                "seed: 42",
                f"cases: {pairs_path.as_posix()}",
                "split: eval",
                f"style_reference: {(ROOT / 'stylometry/data/style_reference.json').as_posix()}",
                f"lexicon: {(ROOT / 'stylometry/data/lexicon.json').as_posix()}",
                f"gazetteer: {(ROOT / 'corpus/gazetteer.json').as_posix()}",
                "judge_model: deepseek-v4-pro",
                f"judge_prompt: {(ROOT / 'eval/prompts/pairwise_judge.txt').as_posix()}",
                "opponent: vernacular",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "config": config,
        "generations": generations,
        "manifest": tmp_path / "agent-run-manifest.json",
        "pairs": pairs_path,
        "plan": plan_path,
        "reports": tmp_path / "reports",
    }


def _argv(paths: dict[str, Path]) -> list[str]:
    return [
        *("--plan", str(paths["plan"])),
        *("--pairs", str(paths["pairs"])),
        *("--config", str(paths["config"])),
        *("--baselines-dir", str(paths["generations"])),
        *("--adapter", str(paths["generations"].parent / "sutong-v2" / "adapter")),
        *("--output-dir", str(paths["generations"])),
        *("--manifest", str(paths["manifest"])),
    ]


def _run(paths: dict[str, Path], generator: _RevisionGenerator) -> None:
    code = main(
        _argv(paths),
        generator=generator,
        counter=_FixedCounter(),
        scorer=FakeAgentScorer(always=1.0),
    )
    assert code == 0


def _rows(paths: dict[str, Path], key: str) -> dict[str, dict[str, object]]:
    text = (paths["generations"] / f"{key}.jsonl").read_text(encoding="utf-8")
    return {row["id"]: row for row in map(json.loads, text.splitlines())}


def test_four_arms_take_round_one_from_the_sweep_files(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    generator = _RevisionGenerator()
    _run(paths, generator)
    # One revision prompt per arm for e1. Round one is never decoded again.
    assert len(generator.calls) == 4
    first_rounds = {
        json.dumps(build_prompt(DRIFTING, []), ensure_ascii=False),
        json.dumps(build_prompt(CLEAN, []), ensure_ascii=False),
    }
    assert all(json.dumps(call, ensure_ascii=False) not in first_rounds for call in generator.calls)
    for arm in arms():
        rows = _rows(paths, arm.key)
        assert set(rows) == {"e1", "e2"}
        drifting, clean = rows["e1"], rows["e2"]
        assert drifting["pipeline"] == "agent"
        assert drifting["output"] == REPAIRED
        assert drifting["selected_round"] == 2
        assert drifting["total_rounds"] == 2
        assert drifting["model_calls"] == 1
        assert drifting["trace"]
        assert clean["output"] == FIRST_CLEAN
        assert clean["model_calls"] == 0
        assert clean["total_rounds"] == 1
        assert clean["termination"] == "accepted"
        expected_ids = ["t2", "t1"] if arm.source == "balanced-k2" else []
        assert drifting["exemplar_ids"] == expected_ids
    text = paths["manifest"].read_text(encoding="utf-8")
    assert MARKER not in text
    assert REPAIRED not in text
    manifest = json.loads(text)
    assert manifest["primary_arm"] == PRIMARY_ARM
    assert set(manifest["arms"]) == {arm.key for arm in arms()}
    record = manifest["arms"][PRIMARY_ARM]
    assert record["n"] == 2
    assert record["model_calls"] == 1
    assert record["first_round_violation_cases"] == 1
    assert record["rounds"]["total_rounds"] == {"1": 1, "2": 1, "3": 0}
    assert record["prompt_tokens"]["revision"]["n"] == 1.0
    assert set(manifest["baselines"]) == {"retrieval-balanced-k2", "retrieval-k0"}


def test_missing_first_round_sha_stops_before_any_generation(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    baseline = paths["generations"] / "retrieval-k0.jsonl"
    rows = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines()]
    rows[0]["prompt_sha256"] = "0" * 64
    baseline.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    generator = _RevisionGenerator()
    with pytest.raises(SystemExit, match="第一轮 prompt"):
        _run(paths, generator)
    assert generator.calls == []
    assert list(paths["generations"].glob("agent-*.jsonl")) == []


def test_persistent_violation_decodes_at_most_twice(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    generator = _RevisionGenerator(["戊走出门去", "戊出门了。"])
    _run(paths, generator)
    row = _rows(paths, PRIMARY_ARM)["e1"]
    assert row["model_calls"] == 2
    assert row["total_rounds"] == 3
    assert row["termination"] == "max_rounds"
    assert _rows(paths, PRIMARY_ARM)["e2"]["model_calls"] == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["arms"][PRIMARY_ARM]["rounds"]["total_rounds"] == {"1": 1, "2": 0, "3": 1}
    assert manifest["arms"][PRIMARY_ARM]["rounds"]["termination"]["max_rounds"] == 1


def test_the_two_feedback_formats_send_different_second_rounds(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    generator = _RevisionGenerator()
    _run(paths, generator)
    violations = [Violation(kind="numeral_missing", expected="12块", actual=None)]
    exemplars = [
        Exemplar(id="t1", vernacular="甲去井边", original="井边有一个人"),
        Exemplar(id="t2", vernacular="乙买了米", original="米放在门口"),
    ]
    expected = [
        build_revision_messages(
            DRIFTING,
            exemplars if arm.source == "balanced-k2" else [],
            previous_output=FIRST_DRIFTING,
            violations=violations,
            feedback_format=arm.feedback_format,
        )
        for arm in arms()
    ]
    assert generator.calls == expected
    followup, restate = generator.calls[0], generator.calls[1]
    assert followup[-2] == {"role": "assistant", "content": FIRST_DRIFTING}
    assert followup[-1]["content"].startswith(FOLLOWUP_LEAD)
    assert restate[-1]["content"].startswith(DRIFTING + "\n\n" + RESTATE_LEAD)
    assert all(message["content"] != FIRST_DRIFTING for message in restate)
    assert len(followup) == len(restate) + 2


def test_restart_skips_finished_cases(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    _run(paths, _RevisionGenerator())
    primary = paths["generations"] / f"{PRIMARY_ARM}.jsonl"
    other = paths["generations"] / "agent-k0-restate.jsonl"
    kept = other.read_bytes()
    first_line = primary.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line)["id"] == "e1"
    primary.write_text(first_line + "\n", encoding="utf-8")
    again = _RevisionGenerator()
    _run(paths, again)
    # e1 is already there; the missing e2 has no violation and needs no decoding.
    assert again.calls == []
    assert primary.read_text(encoding="utf-8").splitlines()[0] == first_line
    assert set(_rows(paths, PRIMARY_ARM)) == {"e1", "e2"}
    assert other.read_bytes() == kept
    primary.write_text("", encoding="utf-8")
    third = _RevisionGenerator()
    _run(paths, third)
    assert len(third.calls) == 1


def test_over_budget_revision_names_the_sample(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    with pytest.raises(SystemExit, match="e1"):
        main(
            _argv(paths),
            generator=_RevisionGenerator(),
            counter=_HugeCounter(),
            scorer=FakeAgentScorer(always=1.0),
        )
    assert not paths["manifest"].exists()


def test_run_script_loads_without_the_judge_stack(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    blocked = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['openai'] = None\n"
            "sys.modules['sacrebleu'] = None\n"
            "import scripts.run_agent_eval\n"
            "assert 'eval.judge' not in sys.modules\n"
            "assert 'eval.run_eval' not in sys.modules\n",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert blocked.returncode == 0, blocked.stderr
    paths = _world(tmp_path)
    argv = ", ".join(repr(item) for item in ["--dry-run", *_argv(paths)])
    dry = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from scripts.run_agent_eval import main\n"
            f"main([{argv}])\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'unsloth' not in sys.modules\n"
            "assert 'transformers' not in sys.modules\n",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert f'"{PRIMARY_ARM}": 1' in dry.stdout
    assert '"agent-k0-restate": 2' in dry.stdout
    assert MARKER not in dry.stdout
    assert list(paths["generations"].glob("agent-*.jsonl")) == []


def test_default_scorer_is_fitted_on_the_train_rows(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    assert main(_argv(paths), generator=_RevisionGenerator(), counter=_FixedCounter()) == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["scorer"]["kind"] == "predictor_b"
    assert manifest["scorer"]["n_train"] == 3


def test_summary_is_deterministic_and_free_of_prose(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    _run(paths, _RevisionGenerator())
    config = load_config(paths["config"]).model_copy(update={"pipeline": "retrieval"})
    for key in ("retrieval-balanced-k2", "retrieval-k0"):
        evaluate(
            config,
            paths["generations"] / f"{key}.jsonl",
            key,
            skip_judge=True,
            report_path=paths["reports"] / f"{key}.json",
            cache_dir=tmp_path / "cache",
        )
    summary_path = paths["reports"] / "agent-eval.json"
    markdown_path = paths["reports"] / "agent-eval.md"
    argv = [
        *("--generations-dir", str(paths["generations"])),
        *("--pairs", str(paths["pairs"])),
        *("--config", str(paths["config"])),
        *("--reports-dir", str(paths["reports"])),
        *("--summary", str(summary_path)),
        *("--markdown", str(markdown_path)),
        *("--cache-dir", str(tmp_path / "cache")),
    ]
    first = eval_main(argv, timestamp="2026-10-07T00:00:00Z")
    second = eval_main(argv, timestamp="2026-10-07T00:00:00Z")
    assert first == second
    assert len(list(paths["reports"].glob("agent-*-*.json"))) == 4
    for path in (summary_path, markdown_path):
        text = path.read_text(encoding="utf-8")
        assert MARKER not in text
        assert REPAIRED not in text
        assert FIRST_DRIFTING not in text
    assert first["human_spot_check"] == "pending"
    primary = first["primary"]
    assert isinstance(primary, dict)
    assert primary["arm"] == PRIMARY_ARM
    versus = primary["vs_control"]
    assert versus["baseline"] == "retrieval-balanced-k2"
    assert versus["changed_cases"] == 1
    assert set(versus["optimized"]) == {"entity_recall", "numeral_recall", "hallucination_rate"}
    assert versus["optimized"]["numeral_recall"]["mean_diff"] == pytest.approx(0.5)
    assert versus["unoptimized"]["copy_ratio"]["conclusion"] is None
    assert primary["vs_k0"]["baseline"] == "retrieval-k0"
    arm_blocks = first["arms"]
    exploratory = first["exploratory"]
    assert isinstance(arm_blocks, list)
    assert isinstance(exploratory, dict)
    blocks = {block["name"]: block for block in arm_blocks}
    main_block = blocks[PRIMARY_ARM]
    assert main_block["role"] == "主臂"
    assert main_block["first_round_violation_cases"] == 1
    assert main_block["rounds"]["total_rounds"] == {"1": 1, "2": 1, "3": 0}
    numeral = main_block["violation_transitions"]["numeral_missing"]
    assert numeral == {
        "final": 0,
        "final_cases": 0,
        "first": 1,
        "first_cases": 1,
        "fixed": 1,
        "introduced": 0,
    }
    assert set(exploratory) == {arm.key for arm in arms()} - {PRIMARY_ARM}
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "agent 直接优化的指标" in markdown
    assert "agent 没有优化的指标" in markdown
    assert "同一套规则" in markdown


def _agent_row(
    total: int,
    selected: int | None,
    termination: str,
    rounds: list[list[tuple[str, str | None, str | None]]],
) -> dict[str, object]:
    return {
        "id": "x",
        "rounds": [
            {
                "violations": [
                    {"kind": kind, "expected": expected, "actual": actual}
                    for kind, expected, actual in items
                ]
            }
            for items in rounds
        ],
        "selected_round": selected,
        "termination": termination,
        "total_rounds": total,
    }


def test_round_distribution_counts_every_bucket() -> None:
    rows = [
        _agent_row(1, 1, "accepted", [[]]),
        _agent_row(2, 2, "accepted", [[("title", "太医", "宫监")], []]),
        _agent_row(3, 1, "max_rounds", [[("title", "太医", None)]] * 3),
        _agent_row(1, None, "fallback", []),
    ]
    assert round_distribution(rows) == {
        "selected_round": {"1": 2, "2": 1, "3": 0, "none": 1},
        "termination": {"accepted": 2, "fallback": 1, "max_rounds": 1, "recursion_limit": 0},
        "total_rounds": {"1": 2, "2": 1, "3": 1},
    }
    assert first_round_violation_cases(rows) == 2


def test_violation_transitions_separate_fixed_from_introduced() -> None:
    rows = [
        # The numeral is fixed, but the revision invents a name.
        _agent_row(
            2,
            2,
            "accepted",
            [
                [("numeral_missing", "12块", None), ("title", "太医", "宫监")],
                [("title", "太医", "宫监"), ("entity_hallucination", None, "北京")],
            ],
        ),
        # Later rounds are worse, so round one is kept and nothing moves.
        _agent_row(
            3,
            1,
            "max_rounds",
            [
                [("numeral_missing", "三斤", None)],
                [("numeral_missing", "三斤", None), ("entity_missing", "颂莲", None)],
                [("numeral_missing", "三斤", None), ("entity_missing", "颂莲", None)],
            ],
        ),
    ]
    table = violation_transitions(rows)
    assert table["numeral_missing"] == {
        "final": 1,
        "final_cases": 1,
        "first": 2,
        "first_cases": 2,
        "fixed": 1,
        "introduced": 0,
    }
    assert table["entity_hallucination"]["introduced"] == 1
    assert table["entity_hallucination"]["first"] == 0
    assert table["title"]["fixed"] == 0
    assert table["title"]["final"] == 1
    assert table["entity_missing"] == {
        "final": 0,
        "final_cases": 0,
        "first": 0,
        "first_cases": 0,
        "fixed": 0,
        "introduced": 0,
    }


def _report(numeral: list[float], hallucination: list[float]) -> dict[str, object]:
    per_case = [
        {
            "id": f"c{index}",
            "metrics": {
                "cjk_numeral_z": 0.0,
                "copy_ratio": 0.5,
                "entity_recall": 1.0,
                "hallucination_rate": hallucination[index],
                "length_ratio": 1.0,
                "numeral_recall": numeral[index],
                "profile_gap_per_case": 0.4,
                "style_distance": 1.0,
            },
            "output_hash": f"sha256:{numeral[index]}-{hallucination[index]}",
        }
        for index in range(len(numeral))
    ]
    return {
        "aggregate": {"profile_gap": 0.2},
        "per_case": per_case,
        "profile_mean_delta": {"cjk_numeral_ratio": -0.1},
    }


def test_conclusions_follow_the_interval_in_both_directions() -> None:
    order = [f"c{index}" for index in range(6)]
    baseline = _report([0.0] * 6, [0.5] * 6)
    better = compare(_report([1.0] * 6, [0.5] * 6), baseline, order)
    optimized = better["optimized"]
    assert isinstance(optimized, dict)
    assert optimized["numeral_recall"]["conclusion"] == "改善"
    assert optimized["hallucination_rate"]["conclusion"] == "未检出差异"
    assert optimized["entity_recall"]["conclusion"] == "未检出差异"
    assert better["changed_cases"] == 6
    assert exit_condition(better)["met"] is False
    worse = compare(_report([0.0] * 6, [1.0] * 6), baseline, order)
    assert exit_condition(worse)["hallucination_rate_conclusion"] == "恶化"
    assert exit_condition(worse)["met"] is False
    lower = compare(_report([0.0] * 6, [0.0] * 6), baseline, order)
    assert exit_condition(lower)["met"] is True
    mixed = compare(_report([1.0, 0.0] * 3, [0.5] * 6), _report([0.0, 1.0] * 3, [0.5] * 6), order)
    mixed_optimized = mixed["optimized"]
    assert isinstance(mixed_optimized, dict)
    assert mixed_optimized["numeral_recall"]["conclusion"] == "未检出差异"


def test_feedback_echo_flags_outputs_that_copy_the_feedback_back() -> None:
    def row(doc_id: str, selected: int, outputs: list[str]) -> dict[str, object]:
        missing = [{"kind": "numeral_missing", "expected": "12块", "actual": None}]
        return {
            "id": doc_id,
            "rounds": [
                {"output": text, "violations": missing if index == 0 else []}
                for index, text in enumerate(outputs)
            ],
            "selected_round": selected,
        }

    rows = [
        # A real fix: the number is in the prose, not in quoted feedback.
        row("a", 2, ["戊出门了", "戊带着十二块钱出门"]),
        # The prose is unchanged and the feedback is appended, so the verifier is fooled.
        row("b", 2, ["戊出门了", f"戊出门了\n{RESTATE_LEAD}\n- 输入里的数值「12块」必须保留"]),
        # A paraphrased echo still carries the quoted fragment.
        row("c", 3, ["戊出门了", FOLLOWUP_LEAD, "戊出门了。输入中的数字「12块」要保留"]),
        # An echo that was not selected counts as a round, not as a final output.
        row("d", 1, ["戊出门了", FOLLOWUP_LEAD]),
    ]
    assert feedback_echo(rows) == {
        "final_cases": 2,
        "final_ids": ["b", "c"],
        "revision_rounds": 4,
        "revision_rounds_total": 5,
        "second_round": 3,
        "second_round_total": 4,
    }
