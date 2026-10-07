"""Real verifier / scorer wiring for T2.2. No GPU, no network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.config import AgentConfig
from agent.graph import run_agent
from agent.nodes import default_route
from agent.scorer import PredictorScorer, fit_train_predictor
from agent.state import MAX_GENERATIONS, AgentState
from agent.verifier import FidelityVerifier
from eval.fidelity import assess
from eval.metrics import entity_recall, hallucination_rate, numeral_recall
from retrieval.prompt import Exemplar
from scripts.validate_agent_scorer import main as validate_main
from stylometry.distance import StyleReference
from stylometry.lexicon import LiteraryLexicon
from tests.fakes import (
    FakeAgentGenerator,
    FakeAgentRetriever,
    FakeAgentScorer,
    FakeStylePredictor,
)

GAZETTEER = {"颂莲", "陈佐千"}


class EmptyTagger:
    def entities(self, text: str) -> set[str]:
        return set()


def test_four_violation_kinds_and_surfaces() -> None:
    verifier = FidelityVerifier(GAZETTEER, tagger=EmptyTagger())
    missing_entity = verifier.verify("颂莲坐着", "屋里安静")
    assert any(item.kind == "entity_missing" for item in missing_entity)
    for item in missing_entity:
        if item.kind == "entity_missing":
            assert item.expected is not None
            assert item.expected in "颂莲坐着"

    missing_numeral = verifier.verify("万人大军", "十万铁骑")
    assert any(item.kind == "numeral_missing" for item in missing_numeral)
    for item in missing_numeral:
        if item.kind == "numeral_missing":
            assert item.expected is not None
            assert item.expected in "万人大军"

    hallucinated = verifier.verify("屋里安静", "颂莲来了")
    assert any(item.kind == "entity_hallucination" for item in hallucinated)
    for item in hallucinated:
        if item.kind == "entity_hallucination":
            assert item.actual is not None
            assert item.actual in "颂莲来了"

    title = verifier.verify("太医来了", "宫监来了")
    assert any(item.kind == "title" for item in title)
    for item in title:
        if item.kind == "title":
            assert item.expected is not None and item.expected in "太医来了"
            assert item.actual is not None and item.actual in "宫监来了"

    fullwidth = verifier.verify("三辆马车", "３辆马车")
    assert fullwidth == []


def test_violations_empty_iff_metrics_perfect() -> None:
    cases = [
        ("颂莲坐着", "颂莲站着", True),
        ("万人大军", "十万铁骑", False),
        ("太医来了", "宫监来了", False),
        ("屋里安静", "颂莲来了", False),
        ("颂莲坐着", "屋里安静", False),
        ("三辆马车", "３辆马车", True),
    ]
    for source, output, expect_empty in cases:
        report = assess(source, output, GAZETTEER, EmptyTagger())
        metrics_ok = (
            report.entity_recall == 1.0
            and report.numeral_recall == 1.0
            and report.hallucination_rate == 0.0
            and not any(item.kind == "title" for item in report.violations)
        )
        assert metrics_ok == (report.violations == [])
        assert (report.violations == []) == expect_empty
        # Keep the metric helpers imported and exercised beside assess().
        assert entity_recall(set(), set()) == 1.0
        assert numeral_recall(set(), set()) == 1.0
        assert hallucination_rate(set(), set()) == 0.0


def test_route_revise_with_real_verifier() -> None:
    result = run_agent(
        "太医来了",
        retriever=FakeAgentRetriever([[]]),
        generator=FakeAgentGenerator(["宫监来了", "太医来了"]),
        verifier=FidelityVerifier(set(), tagger=EmptyTagger()),
        scorer=FakeAgentScorer(always=0.0),
        config=AgentConfig(),
    )
    assert result.total_rounds == 2
    assert result.output == "太医来了"
    assert result.termination == "accepted"


def test_route_re_retrieve_when_threshold_set() -> None:
    retriever = FakeAgentRetriever(
        [
            [Exemplar(id="a", vernacular="甲", original="甲原文")],
            [Exemplar(id="b", vernacular="乙", original="乙原文")],
        ]
    )
    result = run_agent(
        "颂莲坐着",
        retriever=retriever,
        generator=FakeAgentGenerator(["颂莲站着", "颂莲站着"]),
        verifier=FidelityVerifier(GAZETTEER, tagger=EmptyTagger()),
        scorer=FakeAgentScorer(always=-1.0),
        config=AgentConfig(re_retrieve_score_threshold=-0.5),
    )
    assert len(retriever.calls) == 2
    assert result.re_retrieved is True


def test_route_accept_when_threshold_null_despite_low_score() -> None:
    retriever = FakeAgentRetriever([[]])
    result = run_agent(
        "颂莲坐着",
        retriever=retriever,
        generator=FakeAgentGenerator(["颂莲站着"]),
        verifier=FidelityVerifier(GAZETTEER, tagger=EmptyTagger()),
        scorer=FakeAgentScorer(always=-1.0),
        config=AgentConfig(re_retrieve_score_threshold=None),
    )
    assert len(retriever.calls) == 1
    assert result.re_retrieved is False
    assert result.termination == "accepted"


def test_fit_train_predictor_and_matrix_target() -> None:
    lexicon = LiteraryLexicon(literary={}, colloquial={})
    reference = StyleReference(
        mean=np.zeros(20, dtype=np.float64),
        std=np.ones(20, dtype=np.float64),
        lexicon=lexicon,
    )
    vernaculars = [f"白话句子{i}。" * 3 for i in range(6)]
    originals = [f"原文字句{i}。" * 3 for i in range(6)]
    works = ["园艺"] * 6
    predictor = fit_train_predictor(reference, vernaculars, originals, works)
    scorer = PredictorScorer(reference, predictor)
    score = scorer.score(vernaculars[0], originals[0])
    assert isinstance(score, float)

    class MatrixPredictor:
        def predict(self, x: np.ndarray) -> np.ndarray:
            row = np.zeros(20, dtype=np.float64)
            return np.vstack([row, row])

    matrix_scorer = PredictorScorer(reference, MatrixPredictor())
    assert isinstance(matrix_scorer.score("甲", "乙"), float)


def test_predictor_scorer_prefers_exact_target() -> None:
    lexicon = LiteraryLexicon(literary={}, colloquial={})
    mean = np.zeros(20, dtype=np.float64)
    std = np.ones(20, dtype=np.float64)
    reference = StyleReference(mean=mean, std=std, lexicon=lexicon)
    target = np.linspace(-0.5, 0.5, 20)
    scorer = PredictorScorer(reference, FakeStylePredictor(target))

    # Craft an "output" whose zscore equals target: extract(output) = mean + std * target = target
    # With mean 0 std 1, zscore = extract(text). We cannot easily craft extract==target without
    # mocking extract; instead compare relative ordering via a fake reference subclass.
    class FixedReference(StyleReference):
        def zscore(self, text: str) -> np.ndarray:
            if text == "match":
                return target.copy()
            return target.copy() + 1.0

    fixed = FixedReference(mean=mean, std=std, lexicon=lexicon)
    scorer = PredictorScorer(fixed, FakeStylePredictor(target))
    assert scorer.score("input", "match") > scorer.score("input", "other")


def test_validate_agent_scorer_tmp_path(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.jsonl"
    rows = [
        {
            "id": "园艺_0001",
            "work": "园艺",
            "idx": 1,
            "vernacular": "孔先生坐在院子里喝茶。",
            "original": "孔先生坐在院子里喝茶。",
            "split": "train",
        },
        {
            "id": "园艺_0002",
            "work": "园艺",
            "idx": 2,
            "vernacular": "孔太太在厨房里洗菜。",
            "original": "孔太太在厨房里洗菜。",
            "split": "train",
        },
        {
            "id": "园艺_0003",
            "work": "园艺",
            "idx": 3,
            "vernacular": "孔先生走过大门口。",
            "original": "孔先生走过大门口。",
            "split": "train",
        },
        {
            "id": "园艺_0004",
            "work": "园艺",
            "idx": 4,
            "vernacular": "孔太太把灯吹灭了。",
            "original": "孔太太把灯吹灭了。",
            "split": "train",
        },
        {
            "id": "园艺_0005",
            "work": "园艺",
            "idx": 5,
            "vernacular": "天黑下来了。",
            "original": "天黑下来了。",
            "split": "train",
        },
        {
            "id": "园艺_0099",
            "work": "园艺",
            "idx": 99,
            "vernacular": "白话一句。",
            "original": "原文一句。",
            "split": "eval",
        },
    ]
    pairs.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    for name in ("base", "lora"):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps(
                {"id": "园艺_0099", "output": f"{name}输出一句。", "model": name, "pipeline": "x"},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    # Minimal style reference + lexicon + config pointing at temp files.
    lexicon_path = tmp_path / "lexicon.json"
    lexicon_path.write_text(
        LiteraryLexicon(literary={}, colloquial={}).model_dump_json(),
        encoding="utf-8",
    )
    style_path = tmp_path / "style.json"
    style_path.write_text(
        json.dumps({"mean": [0.0] * 20, "std": [1.0] * 20}, ensure_ascii=False),
        encoding="utf-8",
    )
    gaz_path = tmp_path / "gaz.json"
    gaz_path.write_text(json.dumps({"words": []}, ensure_ascii=False), encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "园艺_0099",
                "vernacular": "白话一句。",
                "original": "原文一句。",
                "split": "eval",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline: tmp",
                "model: tmp",
                "seed: 42",
                f"cases: {cases_path.as_posix()}",
                f"style_reference: {style_path.as_posix()}",
                f"lexicon: {lexicon_path.as_posix()}",
                f"gazetteer: {gaz_path.as_posix()}",
                "judge_model: none",
                "judge_prompt: none",
                "opponent: vernacular",
                "split: eval",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    validate_main(
        [
            "--pairs",
            str(pairs),
            "--config",
            str(config_path),
            "--base",
            str(tmp_path / "base.jsonl"),
            "--lora",
            str(tmp_path / "lora.jsonl"),
            "--output",
            str(out),
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_eval"] == 1
    assert "candidate_b" in payload


def test_default_route_re_retrieve_branch() -> None:
    state: AgentState = {
        "iter": 1,
        "max_generations": MAX_GENERATIONS,
        "last_violations": [],
        "last_score": -1.0,
        "score_threshold": -0.5,
        "re_retrieved": False,
        "candidates": ["ok"],
    }
    assert default_route(state) == "re_retrieve"
