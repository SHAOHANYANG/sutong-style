import json
from pathlib import Path

from eval.run_eval import load_generations
from scripts.eval_sweep import judge_difference, main, mean_interval, status_targets
from scripts.sweep_topk import main as sweep_main
from scripts.train import Pair

ROOT = Path(__file__).resolve().parents[1]


class _FakeGenerator:
    def generate(self, messages: list[dict[str, str]]) -> str:
        return "假输出甲"


class _FixedCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return 12


def test_interval_direction_follows_the_pre_registration() -> None:
    _mean, low, high = mean_interval([1.0] * 6)
    assert low > 0.0
    assert judge_difference(low, high, "higher") == "改善"
    assert judge_difference(low, high, "lower") == "恶化"
    _mean, low, high = mean_interval([1.0, -1.0] * 6)
    assert low <= 0.0 <= high
    assert judge_difference(low, high, "higher") == "未检出差异"
    assert judge_difference(low, high, "lower") == "未检出差异"


def test_status_targets_use_the_registered_comparisons() -> None:
    met = status_targets(
        {"numeral_recall": 0.92, "hallucination_rate": 0.005, "profile_gap": 0.203}
    )
    missed = status_targets(
        {"numeral_recall": 0.919, "hallucination_rate": 0.006, "profile_gap": 0.204}
    )
    assert met["numeral_recall"]["met"] is True
    assert met["hallucination_rate"]["met"] is True
    assert met["profile_gap"]["met"] is True
    assert missed["numeral_recall"]["met"] is False
    assert missed["hallucination_rate"]["met"] is False
    assert missed["profile_gap"]["met"] is False


def test_extra_generation_fields_do_not_affect_loading(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "exemplar_ids": ["t2", "t1"],
                "fusion_config": "balanced",
                "id": "e1",
                "k": 2,
                "model": "sutong-v2",
                "output": "门开着",
                "pipeline": "retrieval",
                "prompt_sha256": "abc",
                "prompt_tokens": 12,
                "seed": 42,
                "trace": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_generations(path)
    assert loaded["e1"].output == "门开着"
    assert loaded["e1"].pipeline == "retrieval"


def test_summary_is_deterministic_and_names_the_conclusion(tmp_path: Path) -> None:
    def one(doc_id: str, vernacular: str, original: str, split: str) -> Pair:
        return Pair(
            id=doc_id,
            work="自编",
            idx=1,
            vernacular=vernacular,
            original=original,
            split=split,
        )

    pairs = [
        one("t1", "甲去井边", "井边有一个人", "train"),
        one("t2", "乙买了米", "米放在门口", "train"),
        one("t3", "丙看着天", "天色暗下来", "train"),
        one("e1", "戊要出门", "门开着", "eval"),
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
                    "id": "e1",
                    "work": "自编",
                }
            ]
            for config in ("equal", "balanced", "content", "style")
        }
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "generations"
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(json.dumps({"id": "e1", "output": "假输出甲"}) + "\n", encoding="utf-8")
    sweep_main(
        [
            "--plan",
            str(plan_path),
            "--pairs",
            str(pairs_path),
            "--baseline",
            str(baseline),
            "--adapter",
            str(tmp_path / "sutong-v2" / "adapter"),
            "--output-dir",
            str(output),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
        generator=_FakeGenerator(),
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
    reports = tmp_path / "reports"
    summary_path = reports / "retrieval-sweep.json"
    markdown_path = reports / "retrieval-sweep.md"
    argv = [
        "--generations-dir",
        str(output),
        "--pairs",
        str(pairs_path),
        "--config",
        str(config),
        "--reports-dir",
        str(reports),
        "--summary",
        str(summary_path),
        "--markdown",
        str(markdown_path),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    first = main(argv, timestamp="2026-10-07T00:00:00Z")
    second = main(argv, timestamp="2026-10-07T00:00:00Z")
    assert first == second
    primary = first["primary"]
    assert isinstance(primary, dict)
    metrics = primary["metrics"]
    assert isinstance(metrics, dict)
    entity = metrics["entity_recall"]
    assert isinstance(entity, dict)
    assert entity["conclusion"] == "未检出差异"
    assert "戊要出门" not in summary_path.read_text(encoding="utf-8")
    assert "未检出差异" in markdown_path.read_text(encoding="utf-8")
    assert len(list(reports.glob("retrieval-*.json"))) == 14
