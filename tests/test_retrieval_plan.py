import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from retrieval.dense import save_dense_cache
from retrieval.hybrid import DEFAULT_DEPTH, DEFAULT_RRF_K, FUSION_CONFIGS
from retrieval.types import Document, FusedHit
from scripts.build_retrieval_plan import jaccard, load_plan_config, main, validate_hits
from scripts.split_corpus import CorpusSplit
from scripts.train import Pair

ROOT = Path(__file__).resolve().parents[1]
STYLE_CONFIG = ROOT / "eval" / "configs" / "eval59.yaml"


def _pair(doc_id: str, split: str, vernacular: str, original: str, work: str = "自编") -> Pair:
    return Pair(
        id=doc_id,
        work=work,
        idx=1,
        vernacular=vernacular,
        original=original,
        split=split,
    )


def _write_cache(directory: Path, documents: list[Document]) -> None:
    matrix = np.arange(len(documents) * 4, dtype=np.float64).reshape(len(documents), 4) + 1.0
    save_dense_cache(
        documents,
        matrix,
        model_id="BAAI/bge-m3",
        revision="rev-test",
        max_length=512,
        directory=directory,
    )


def _config(path: Path) -> None:
    source = ROOT / "eval" / "configs" / "retrieval.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("检索配置不是映射")
    payload["embedding_revision"] = "rev-test"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8")


def _layout(tmp_path: Path, train: list[Pair], eval_rows: list[Pair]) -> dict[str, str]:
    pairs_path = tmp_path / "pairs.jsonl"
    split_path = tmp_path / "split.json"
    index_dir = tmp_path / "index"
    query_dir = tmp_path / "query"
    config_path = tmp_path / "retrieval.yaml"
    rows = train + eval_rows
    pairs_path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    split_path.write_text(
        CorpusSplit(
            seed=42,
            eval_ratio=0.08,
            train=[row.id for row in train],
            eval=[row.id for row in eval_rows],
        ).model_dump_json(),
        encoding="utf-8",
    )
    _write_cache(index_dir, [Document(id=row.id, text=row.original) for row in train])
    _write_cache(query_dir, [Document(id=row.id, text=row.vernacular) for row in eval_rows])
    _config(config_path)
    return {
        "pairs": str(pairs_path),
        "split": str(split_path),
        "index": str(index_dir),
        "query": str(query_dir),
        "config": str(config_path),
    }


def _argv(paths: dict[str, str], output: Path) -> list[str]:
    return [
        "--pairs",
        paths["pairs"],
        "--split",
        paths["split"],
        "--config",
        paths["config"],
        "--style-config",
        str(STYLE_CONFIG),
        "--index-dir",
        paths["index"],
        "--query-dir",
        paths["query"],
        "--output",
        str(output),
    ]


def _train() -> list[Pair]:
    return [
        _pair("t1", "train", "甲去井边", "井边有一个人", "北"),
        _pair("t2", "train", "乙买了米", "米放在门口", "南"),
        _pair("t3", "train", "丙看着天", "天色暗下来", "北"),
    ]


def test_plan_report_has_four_configs_and_three_exemplars(tmp_path: Path) -> None:
    paths = _layout(tmp_path, _train(), [_pair("e1", "eval", "丁要出门", "门开着")])
    output = tmp_path / "plan.json"
    main(_argv(paths, output))
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["plans"]) == {"balanced", "content", "equal", "style"}
    assert report["parameters"]["n_eval"] == 1
    assert report["parameters"]["n_train"] == 3
    assert report["parameters"]["primary_config"] == "balanced"
    assert report["parameters"]["primary_k"] == 2
    blob = output.read_text(encoding="utf-8")
    assert "丁要出门" not in blob
    assert "门开着" not in blob
    for name, rows in report["plans"].items():
        assert len(rows) == 1
        exemplars = rows[0]["exemplars"]
        assert [item["rank"] for item in exemplars] == [1, 2, 3]
        assert {item["id"] for item in exemplars} == {"t1", "t2", "t3"}
        assert "k1" in report["summary"][name]
        assert "k2" in report["summary"][name]
        assert "k3" in report["summary"][name]
        assert "profile_gap" in report["summary"][name]["k2"]
        assert "by_work" in report["summary"][name]["k2"]
    content_sources = report["plans"]["content"][0]["exemplars"][0]["sources"]
    style_sources = report["plans"]["style"][0]["exemplars"][0]["sources"]
    assert "style" not in content_sources
    assert set(style_sources) == {"style"}
    assert "balanced|equal" in report["jaccard"]["k2"]
    assert set(report["caches"]["index"]) == {"created_at", "max_length", "model_id", "revision"}


def test_plan_is_deterministic_except_for_the_timestamp(tmp_path: Path) -> None:
    paths = _layout(tmp_path, _train(), [_pair("e1", "eval", "丁要出门", "门开着")])
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    main(_argv(paths, first))
    main(_argv(paths, second))
    left = json.loads(first.read_text(encoding="utf-8"))
    right = json.loads(second.read_text(encoding="utf-8"))
    left.pop("timestamp")
    right.pop("timestamp")
    assert left == right


def test_missing_query_cache_names_the_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-query-cache"
    with pytest.raises(SystemExit, match="missing-query-cache"):
        main(["--query-dir", str(missing), "--output", str(tmp_path / "out.json")])


def test_fewer_than_three_exemplars_stops_the_script(tmp_path: Path) -> None:
    train = [
        _pair("t1", "train", "甲去井边", "井边有一个人", "北"),
        _pair("t2", "train", "乙买了米", "米放在门口", "北"),
    ]
    paths = _layout(tmp_path, train, [_pair("e1", "eval", "丁要出门", "门开着")])
    with pytest.raises(SystemExit, match="e1"):
        main(_argv(paths, tmp_path / "out.json"))


def test_plan_rules_stop_the_run() -> None:
    ok = [
        FusedHit(id="t1", rank=1, score=0.2, sources={"bm25": 1, "dense": 1}),
        FusedHit(id="t2", rank=2, score=0.1, sources={"bm25": 2}),
        FusedHit(id="t3", rank=3, score=0.05, sources={"dense": 1}),
    ]
    leaked = [FusedHit(id="e9", rank=1, score=1.0, sources={"bm25": 1}), *ok[1:]]
    with pytest.raises(SystemExit, match=r"split\.eval"):
        validate_hits("equal", "e1", leaked, {"e9"})
    own = [FusedHit(id="e1", rank=1, score=1.0, sources={"bm25": 1}), *ok[1:]]
    with pytest.raises(SystemExit, match="查询自己"):
        validate_hits("equal", "e1", own, set())
    with pytest.raises(SystemExit, match="不足"):
        validate_hits("equal", "e1", ok[:2], set())
    with pytest.raises(SystemExit, match="content"):
        validate_hits(
            "content",
            "e1",
            [FusedHit(id="t1", rank=1, score=1.0, sources={"style": 1}), *ok[1:]],
            set(),
        )
    with pytest.raises(SystemExit, match="style"):
        validate_hits("style", "e1", ok, set())


def test_jaccard_of_identical_disjoint_and_half_overlap() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b", "c"}, {"a", "b", "d"}) == 0.5


def test_retrieval_yaml_matches_fusion_configs() -> None:
    config = load_plan_config(ROOT / "eval" / "configs" / "retrieval.yaml")
    assert config.depth == DEFAULT_DEPTH
    assert config.rrf_k == DEFAULT_RRF_K
    assert config.primary_config == "balanced"
    assert config.primary_k == 2
    assert config.embedding_revision == "5617a9f61b028005a4858fdac845db406aefb181"
    got = {
        name: {"bm25": item.bm25, "dense": item.dense, "style": item.style}
        for name, item in config.configs.items()
    }
    assert got == FUSION_CONFIGS
