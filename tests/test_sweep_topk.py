import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.generate import generate_one, greedy_decode
from scripts.sweep_topk import group_key, jobs, main
from scripts.train import SYSTEM_PROMPT, Pair, build_messages

ROOT = Path(__file__).resolve().parents[1]


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return "假输出甲"


class _FixedCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return 12


class _HugeCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return 10_000


def _pair(doc_id: str, split: str, vernacular: str, original: str) -> Pair:
    return Pair(
        id=doc_id,
        work="自编",
        idx=1,
        vernacular=vernacular,
        original=original,
        split=split,
    )


def _world(tmp_path: Path, ranks: dict[str, list[str]]) -> dict[str, Path]:
    pairs = [
        _pair("t1", "train", "甲去井边", "井边有一个人"),
        _pair("t2", "train", "乙买了米", "米放在门口"),
        _pair("t3", "train", "丙看着天", "天色暗下来"),
        _pair("t4", "train", "丁关了门", "门已经关上"),
        _pair("e1", "eval", "戊要出门", "门开着"),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(row.model_dump_json() + "\n" for row in pairs), encoding="utf-8")
    plans = {
        config: [
            {
                "exemplars": [
                    {"id": item, "rank": rank, "score": 0.2, "sources": {"bm25": rank}}
                    for rank, item in enumerate(ids, start=1)
                ],
                "id": "e1",
                "work": "自编",
            }
        ]
        for config, ids in ranks.items()
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plans": plans}, ensure_ascii=False), encoding="utf-8")
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        json.dumps({"id": "e1", "output": "假输出甲"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "baseline": baseline,
        "manifest": tmp_path / "manifest.json",
        "output": tmp_path / "out",
        "pairs": pairs_path,
        "plan": plan_path,
    }


def _same_ranks() -> dict[str, list[str]]:
    return {name: ["t1", "t2", "t3"] for name in ("equal", "balanced", "content", "style")}


def _argv(paths: dict[str, Path]) -> list[str]:
    adapter = paths["output"].parent / "sutong-v2" / "adapter"
    return [
        "--plan",
        str(paths["plan"]),
        "--pairs",
        str(paths["pairs"]),
        "--baseline",
        str(paths["baseline"]),
        "--adapter",
        str(adapter),
        "--output-dir",
        str(paths["output"]),
        "--manifest",
        str(paths["manifest"]),
    ]


def test_generate_one_uses_the_shared_decoder() -> None:
    assert "greedy_decode" in inspect.getsource(generate_one)
    assert greedy_decode.__module__ == "scripts.generate"


def test_sweep_writes_thirteen_groups_and_prompt_order(tmp_path: Path) -> None:
    paths = _world(tmp_path, _same_ranks())
    generator = _FakeGenerator()
    assert main(_argv(paths), generator=generator, counter=_FixedCounter()) == 0
    assert len(list(paths["output"].glob("*.jsonl"))) == 13
    assert generator.calls[0] == build_messages("戊要出门", None)
    assert generator.calls[0][0]["content"] == SYSTEM_PROMPT
    k0_path = paths["output"] / "retrieval-k0.jsonl"
    k0 = json.loads(k0_path.read_text(encoding="utf-8"))
    assert k0["pipeline"] == "retrieval"
    assert k0["fusion_config"] is None
    assert k0["k"] == 0
    assert k0["exemplar_ids"] == []
    assert k0["prompt_tokens"] == 12
    assert k0["model"] == "sutong-v2"
    equal_path = paths["output"] / "retrieval-equal-k2.jsonl"
    equal = json.loads(equal_path.read_text(encoding="utf-8"))
    assert equal["exemplar_ids"] == ["t2", "t1"]
    assert equal["fusion_config"] == "equal"
    assert equal["k"] == 2
    assert "戊要出门" not in paths["manifest"].read_text(encoding="utf-8")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["k0_check"]["passed"] is True
    assert set(manifest["groups"]) == {group_key(config, k) for config, k in jobs()}


def test_identical_exemplars_are_generated_once(tmp_path: Path) -> None:
    paths = _world(tmp_path, _same_ranks())
    generator = _FakeGenerator()
    main(_argv(paths), generator=generator, counter=_FixedCounter())
    assert len(generator.calls) == 4


def test_k0_mismatch_stops_before_the_other_groups(tmp_path: Path) -> None:
    paths = _world(tmp_path, _same_ranks())
    paths["baseline"].write_text(
        json.dumps({"id": "e1", "output": "另一段"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="e1"):
        main(_argv(paths), generator=_FakeGenerator(), counter=_FixedCounter())
    names = sorted(path.name for path in paths["output"].glob("*.jsonl"))
    assert names == ["retrieval-k0.jsonl"]


def test_resume_skips_existing_rows_and_a_changed_plan_fails(tmp_path: Path) -> None:
    ranks = {
        "equal": ["t1", "t2", "t3"],
        "balanced": ["t2", "t3", "t1"],
        "content": ["t3", "t1", "t2"],
        "style": ["t1", "t3", "t2"],
    }
    paths = _world(tmp_path, ranks)
    main(_argv(paths), generator=_FakeGenerator(), counter=_FixedCounter())
    kept = (paths["output"] / "retrieval-equal-k1.jsonl").read_bytes()
    (paths["output"] / "retrieval-style-k3.jsonl").unlink()
    second = _FakeGenerator()
    main(_argv(paths), generator=second, counter=_FixedCounter())
    assert len(second.calls) == 1
    assert (paths["output"] / "retrieval-equal-k1.jsonl").read_bytes() == kept
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan["plans"]["equal"][0]["exemplars"][0]["id"] = "t4"
    paths["plan"].write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="prompt_sha256"):
        main(_argv(paths), generator=_FakeGenerator(), counter=_FixedCounter())


def test_over_budget_names_the_sample(tmp_path: Path) -> None:
    paths = _world(tmp_path, _same_ranks())
    with pytest.raises(SystemExit, match="e1"):
        main(_argv(paths), generator=_FakeGenerator(), counter=_HugeCounter())
    assert list(paths["output"].glob("*.jsonl")) == []


def test_sweep_imports_neither_eval_stack_nor_the_model(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    banned = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.sweep_topk, sys\n"
            "roots = {name.split('.')[0] for name in sys.modules}\n"
            "assert not roots & {'eval', 'stylometry', 'cn2an', 'openai'}",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert banned.returncode == 0, banned.stderr
    paths = _world(tmp_path, _same_ranks())
    argv = ", ".join(repr(item) for item in ["--dry-run", *_argv(paths)])
    dry = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from scripts.sweep_topk import main\n"
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
    assert '"unique_prompts": 4' in dry.stdout
    assert '"rows": 13' in dry.stdout
