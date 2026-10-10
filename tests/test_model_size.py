import argparse
import json
from pathlib import Path

import pytest

from eval.run_eval import evaluate, load_config
from scripts.eval_model_size import NEAR_COPY, compare, main, probe_block
from scripts.generate import model_name
from scripts.generate import parse_args as generate_args
from scripts.run_model_size import TAGS, plan_steps
from scripts.run_model_size import main as run_main
from scripts.train import (
    BASE_MODEL,
    EXPECTED_TRAINABLE,
    EXPECTED_TRAINABLE_BY_MODEL,
    TARGET_MODULES,
    Pair,
    load_pairs,
    parse_args,
    resolve_config,
)

ROOT = Path(__file__).resolve().parents[1]
BIG = "Qwen/Qwen2.5-14B-Instruct"
MARKER = "独特标记戊己"


def _lora_parameters(hidden: int, mlp: int, layers: int, kv_heads: int, head: int = 128) -> int:
    """r = 16 on q, k, v, o and the three MLP projections of every layer."""
    kv = kv_heads * head
    return 16 * (2 * 2 * hidden + 2 * (hidden + kv) + 3 * (hidden + mlp)) * layers


def test_expected_trainable_counts_follow_the_published_shapes() -> None:
    assert len(TARGET_MODULES) == 7
    assert _lora_parameters(2048, 11008, 36, 2) == EXPECTED_TRAINABLE
    assert (
        _lora_parameters(3584, 18944, 28, 4)
        == EXPECTED_TRAINABLE_BY_MODEL["Qwen/Qwen2.5-7B-Instruct"]
    )
    assert _lora_parameters(5120, 13824, 48, 8) == EXPECTED_TRAINABLE_BY_MODEL[BIG] == 68_812_800


def test_base_model_changes_only_the_base_and_its_parameter_count() -> None:
    default = resolve_config(parse_args([]))
    big = resolve_config(parse_args(["--base-model", BIG]))
    assert default.base_model == BASE_MODEL
    assert default.expected_trainable == EXPECTED_TRAINABLE
    assert big.base_model == BIG
    assert big.expected_trainable == 68_812_800
    same = {"base_model", "expected_trainable"}
    assert default.model_dump(exclude=same) == big.model_dump(exclude=same)
    with pytest.raises(SystemExit):
        parse_args(["--base-model", "Qwen/Qwen2.5-72B-Instruct"])


def test_generation_rows_name_a_non_default_base() -> None:
    assert model_name(None) == "base"
    assert model_name(None, BIG) == "base:" + BIG
    assert model_name(Path("adapters/sutong-v2-14b/adapter"), BIG) == "sutong-v2-14b"
    assert generate_args(["--run-id", "x"]).base_model == BASE_MODEL


def test_plan_trains_once_then_generates_base_and_tuned_on_both_sets() -> None:
    steps = plan_steps(
        base_model=BIG,
        pairs=Path("corpus/pairs.jsonl"),
        probes=Path("eval/probes/modern_inputs.jsonl"),
        adapters_dir=Path("adapters"),
        generations_dir=Path("corpus/generations"),
        python="python",
    )
    assert [step.name for step in steps] == [
        "train",
        "base-14b-eval59",
        "lora-14b-eval59",
        "base-14b-probe",
        "lora-14b-probe",
    ]
    train, base_eval, lora_eval, base_probe, lora_probe = steps
    assert train.command[train.command.index("--base-model") + 1] == BIG
    assert train.command[train.command.index("--run-id") + 1] == "sutong-v2-14b"
    assert train.output == Path("adapters/sutong-v2-14b/loss_curve.json")
    assert "--adapter" not in base_eval.command
    assert base_eval.command[base_eval.command.index("--base-model") + 1] == BIG
    adapter = Path(lora_eval.command[lora_eval.command.index("--adapter") + 1])
    assert adapter == Path("adapters/sutong-v2-14b/adapter")
    assert "--base-model" not in lora_eval.command
    assert base_probe.command[base_probe.command.index("--split") + 1] == "probe"
    assert lora_probe.output == Path("corpus/generations/lora-14b-probe.jsonl")
    assert set(TAGS) == set(EXPECTED_TRAINABLE_BY_MODEL)


def test_dry_run_starts_no_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not start a process")

    monkeypatch.setattr("scripts.run_model_size.subprocess.run", refuse)
    assert run_main(["--dry-run"]) == 0


def test_probe_inputs_are_valid_rows_outside_the_corpus_splits() -> None:
    path = ROOT / "eval/probes/modern_inputs.jsonl"
    rows = load_pairs(path)
    assert len(rows) == 20
    assert len({row.id for row in rows}) == 20
    assert {row.split for row in rows} == {"probe"}
    assert all(row.original == "" and row.vernacular.strip() for row in rows)
    categories = [json.loads(line)["category"] for line in path.read_text("utf-8").splitlines()]
    assert sorted(set(categories)) == ["dialogue", "first_person", "scene", "third_person_short"]


def test_probe_block_counts_copies_and_groups_by_category() -> None:
    inputs = {
        "a": {"id": "a", "vernacular": "甲去井边打水回来了", "category": "x"},
        "b": {"id": "b", "vernacular": "乙买了三斤米放在门口", "category": "x"},
        "c": {"id": "c", "vernacular": "丙抬头看着天不说话", "category": "y"},
    }
    rows = [
        {"id": "a", "output": "甲去井边打水回来了"},
        {"id": "b", "output": "乙买了三斤米放在门口。"},
        {"id": "c", "output": "天色暗了下来，丙一言不发地望着"},
    ]
    block = probe_block(rows, inputs)
    assert block["n"] == 3
    assert block["identical"] == 1
    assert block["near_copies"] == 2
    by_category = block["by_category"]
    assert isinstance(by_category, dict)
    assert by_category["x"]["n"] == 2
    assert by_category["x"]["mean"] >= NEAR_COPY > by_category["y"]["mean"]
    with pytest.raises(SystemExit, match="条数"):
        probe_block(rows[:2], inputs)
    with pytest.raises(SystemExit, match="之外的 id"):
        probe_block([{"id": "zz", "output": "x"}], inputs)


def _report(numeral: list[float], gap: list[float]) -> dict[str, object]:
    return {
        "aggregate": {"profile_gap": sum(gap) / len(gap)},
        "per_case": [
            {
                "id": f"c{index}",
                "metrics": {
                    "copy_ratio": 0.7,
                    "entity_recall": 1.0,
                    "hallucination_rate": 0.0,
                    "numeral_recall": numeral[index],
                    "profile_gap_per_case": gap[index],
                    "style_distance": 1.0,
                },
            }
            for index in range(len(numeral))
        ],
    }


def test_comparison_judges_each_metric_in_its_own_direction() -> None:
    order = [f"c{index}" for index in range(6)]
    block = compare(_report([1.0] * 6, [0.6] * 6), _report([0.5] * 6, [0.4] * 6), order)
    metrics = block["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["numeral_recall"]["conclusion"] == "改善"
    # A larger gap to the reference is worse even though the number went up.
    assert metrics["profile_gap_per_case"]["conclusion"] == "恶化"
    assert metrics["entity_recall"]["conclusion"] == "未检出差异"
    assert metrics["copy_ratio"]["conclusion"] is None
    assert metrics["profile_gap"]["diff"] == pytest.approx(0.2)


def _pair(doc_id: str, split: str, vernacular: str, original: str) -> Pair:
    return Pair(
        id=doc_id, work="自编", idx=1, vernacular=vernacular, original=original, split=split
    )


def test_summary_covers_three_comparisons_and_four_probe_models(tmp_path: Path) -> None:
    pairs = [
        _pair("t1", "train", "甲去井边", "井边有一个人"),
        _pair("e1", "eval", f"戊带着12块钱出门，{MARKER}", "戊揣着十二块钱出了门"),
        _pair("e2", "eval", "己关上了门", "门关上了"),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(row.model_dump_json() + "\n" for row in pairs), encoding="utf-8")
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        json.dumps(
            {"id": "p1", "vernacular": f"庚今天去办事，{MARKER}", "category": "first_person"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    generations = tmp_path / "generations"
    generations.mkdir()
    outputs = {
        "base-eval59": ["戊出门了", "门被己关上了"],
        "lora-eval59": ["戊出门了", "己把门关上"],
        "base-14b-eval59": ["戊带着十二块钱出门", "门被己关上了"],
        "lora-14b-eval59": ["戊揣着十二块钱出了门", "己关上了门"],
    }
    for name, (case_one, case_two) in outputs.items():
        (generations / f"{name}.jsonl").write_text(
            json.dumps({"id": "e1", "output": case_one}, ensure_ascii=False)
            + "\n"
            + json.dumps({"id": "e2", "output": case_two}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    for name, text in {
        "base-3b-probe": f"庚今天去办事，{MARKER}",
        "lora-3b-probe": f"庚今天去办事，{MARKER}",
        "base-14b-probe": "庚这一日出门办事去了",
        "lora-14b-probe": "这一天庚出了门，是去办一桩事情",
    }.items():
        (generations / f"{name}.jsonl").write_text(
            json.dumps({"id": "p1", "output": text}, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline: baseline",
                "model: unknown",
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
    reports = tmp_path / "reports"
    for name in ("base-eval59", "lora-eval59"):
        evaluate(
            load_config(config_path),
            generations / f"{name}.jsonl",
            name,
            skip_judge=True,
            report_path=reports / f"{name}.json",
            cache_dir=tmp_path / "cache",
        )
    argv = [
        *("--generations-dir", str(generations)),
        *("--probes", str(probes)),
        *("--config", str(config_path)),
        *("--reports-dir", str(reports)),
        *("--cache-dir", str(tmp_path / "cache")),
    ]
    first = main(argv, timestamp="2026-10-10T00:00:00Z")
    assert first == main(argv, timestamp="2026-10-10T00:00:00Z")
    assert (reports / "lora-14b-eval59.json").is_file()
    for suffix in (".json", ".md"):
        text = (reports / f"model-size-14b{suffix}").read_text(encoding="utf-8")
        assert MARKER not in text
        assert "戊揣着" not in text
    primary = first["primary"]
    secondary = first["secondary"]
    probe_results = first["probes"]
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)
    assert isinstance(probe_results, dict)
    assert (primary["new"], primary["old"]) == ("lora-14b-eval59", "lora-eval59")
    assert primary["metrics"]["numeral_recall"]["mean_diff"] == pytest.approx(0.5)
    assert set(secondary) == {"fine_tuning_on_new_base", "base_size"}
    assert secondary["base_size"]["old"] == "base-eval59"
    assert list(probe_results) == [
        "base-3b-probe",
        "lora-3b-probe",
        "base-14b-probe",
        "lora-14b-probe",
    ]
    assert probe_results["lora-3b-probe"]["identical"] == 1
    assert probe_results["lora-14b-probe"]["near_copies"] == 0
    markdown = (reports / "model-size-14b.md").read_text(encoding="utf-8")
    assert "探索性" in markdown
    assert "不是人工语料" in markdown
    # A stale report must not be compared with a different generation file.
    (generations / "lora-eval59.jsonl").write_text(
        json.dumps({"id": "e1", "output": "换了一份"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "e2", "output": "己把门关上"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="对不上"):
        main(argv)


def test_argparse_namespace_is_plain() -> None:
    assert isinstance(parse_args(["--dry-run"]), argparse.Namespace)
