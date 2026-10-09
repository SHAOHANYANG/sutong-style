import json
from pathlib import Path

from agent.config import AgentConfig
from agent.graph import run_agent
from agent.prompts import (
    FOLLOWUP_LEAD,
    RESTATE_LEAD,
    SYSTEM_LEAD,
    build_revision_messages,
    feedback_block,
    strip_feedback_echo,
)
from eval.fidelity import Violation
from eval.run_eval import evaluate, load_config
from retrieval.prompt import Exemplar, build_prompt
from scripts.eval_agent import main as eval_main
from scripts.run_agent_eval import V2, arms, main
from scripts.train import SYSTEM_PROMPT
from tests.fakes import FakeAgentGenerator, FakeAgentRetriever, FakeAgentScorer
from tests.test_agent_eval import (
    DRIFTING,
    FIRST_CLEAN,
    FIRST_DRIFTING,
    MARKER,
    REPAIRED,
    _argv,
    _FixedCounter,
    _rows,
    _world,
)

MISSING = [Violation(kind="numeral_missing", expected="12块", actual=None)]
BODY = "戊揣着钱出了门"


class _NeedsTwelve:
    """Violation until the number shows up anywhere in the text, like the real verifier."""

    def verify(self, vernacular: str, output: str) -> list[Violation]:
        return [] if "十二块" in output or "12块" in output else list(MISSING)


def test_strip_cuts_from_a_verbatim_lead() -> None:
    echoed = BODY + "\n" + feedback_block(MISSING, lead=RESTATE_LEAD)
    kept, removed = strip_feedback_echo(echoed, MISSING)
    assert kept == BODY
    assert removed == len(echoed) - len(BODY)
    assert strip_feedback_echo(feedback_block(MISSING, lead=FOLLOWUP_LEAD), MISSING)[0] == ""
    assert strip_feedback_echo(BODY + SYSTEM_LEAD + "尾巴", [])[0] == BODY


def test_strip_cuts_a_reworded_echo_by_its_quoted_fragment() -> None:
    reworded = BODY + "\n请按照下列要求修改输入，保持其他内容不变：\n- 输入中的数字「12块」必须保留"
    assert strip_feedback_echo(reworded, MISSING)[0] == BODY
    without_lead = BODY + "\n- 输入中的数字「12块」必须保留\n再多一行"
    assert strip_feedback_echo(without_lead, MISSING)[0] == BODY


def test_strip_leaves_ordinary_prose_alone() -> None:
    repaired = "戊揣着十二块钱出了门。\n他说：\n“走吧。”"
    assert strip_feedback_echo(repaired, MISSING) == (repaired, 0)
    # Corner quotes around something the feedback never mentioned are not an echo.
    quoted = "门上贴着「福」字，戊揣着十二块钱出了门"
    assert strip_feedback_echo(quoted, MISSING) == (quoted, 0)
    # A fragment without the feedback's quote marks is the fix itself.
    assert strip_feedback_echo("戊带着12块钱出门", MISSING) == ("戊带着12块钱出门", 0)


def test_system_format_keeps_every_user_turn_as_plain_input() -> None:
    exemplars = [Exemplar(id="t1", vernacular="甲去井边", original="井边有一个人")]
    base = build_prompt(DRIFTING, exemplars)
    messages = build_revision_messages(
        DRIFTING,
        exemplars,
        previous_output=FIRST_DRIFTING,
        violations=MISSING,
        feedback_format="system",
    )
    assert messages[1:] == base[1:]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == (
        SYSTEM_PROMPT + "\n\n" + feedback_block(MISSING, lead=SYSTEM_LEAD)
    )
    assert "「12块」" in messages[0]["content"]
    assert all(FIRST_DRIFTING not in message["content"] for message in messages)


def test_guard_verifies_only_the_prose_that_is_left() -> None:
    echoed = BODY + "\n" + feedback_block(MISSING, lead=RESTATE_LEAD)
    off = run_agent(
        DRIFTING,
        retriever=FakeAgentRetriever(),
        generator=FakeAgentGenerator([FIRST_DRIFTING, echoed]),
        verifier=_NeedsTwelve(),
        scorer=FakeAgentScorer(always=1.0),
        config=AgentConfig(feedback_format="restate"),
    )
    # Without the guard the quoted fragment in the copied feedback passes the verifier.
    assert off.output == echoed
    assert off.termination == "accepted"
    on = run_agent(
        DRIFTING,
        retriever=FakeAgentRetriever(),
        generator=FakeAgentGenerator([FIRST_DRIFTING, echoed]),
        verifier=_NeedsTwelve(),
        scorer=FakeAgentScorer(always=1.0),
        config=AgentConfig(feedback_format="restate", echo_guard=True),
    )
    assert [record.output for record in on.rounds] == [FIRST_DRIFTING, BODY, BODY]
    assert [record.violation_count for record in on.rounds] == [1, 1, 1]
    assert on.rounds[0].echo_stripped_chars == 0
    assert on.rounds[1].echo_stripped_chars == len(echoed) - len(BODY)
    assert on.termination == "max_rounds"
    assert on.selected_round == 1
    assert RESTATE_LEAD not in on.output
    payloads = [event.payload for event in on.trace if event.node == "generate"]
    assert [item["echo_stripped_chars"] for item in payloads] == [
        0,
        len(echoed) - len(BODY),
        len(echoed) - len(BODY),
    ]


def test_guard_treats_a_pure_echo_as_a_blank_round() -> None:
    echo = feedback_block(MISSING, lead=FOLLOWUP_LEAD)
    generator = FakeAgentGenerator([FIRST_DRIFTING, echo, FIRST_DRIFTING])
    result = run_agent(
        DRIFTING,
        retriever=FakeAgentRetriever(),
        generator=generator,
        verifier=_NeedsTwelve(),
        scorer=FakeAgentScorer(always=1.0),
        config=AgentConfig(echo_guard=True),
    )
    assert result.rounds[1].output == ""
    assert result.rounds[1].echo_stripped_chars == len(echo)
    # A blank round falls back to the plain first-round prompt (SPEC 4.4).
    assert generator.calls[2] == build_prompt(DRIFTING, [])
    assert result.output == FIRST_DRIFTING
    assert result.selected_round == 1


class _FormatSensitiveGenerator:
    """Copies the feedback back wherever it sits in a user turn; obeys it in the system turn."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        last = messages[-1]["content"]
        if last.startswith(FOLLOWUP_LEAD):
            return last
        if RESTATE_LEAD in last:
            return FIRST_DRIFTING + "。" + last[last.index("\n\n") :]
        if SYSTEM_LEAD in messages[0]["content"]:
            return REPAIRED
        return FIRST_DRIFTING


def test_second_experiment_runs_six_guarded_arms(tmp_path: Path) -> None:
    paths = _world(tmp_path)
    paths["manifest"] = tmp_path / "agent2-run-manifest.json"
    code = main(
        ["--experiment", "v2", *_argv(paths)],
        generator=_FormatSensitiveGenerator(),
        counter=_FixedCounter(),
        scorer=FakeAgentScorer(always=1.0),
    )
    assert code == 0
    keys = [arm.key for arm in arms(V2)]
    assert len(keys) == 6
    assert sorted(path.stem for path in paths["generations"].glob("agent2-*.jsonl")) == sorted(keys)
    assert list(paths["generations"].glob("agent-*.jsonl")) == []
    system = _rows(paths, "agent2-balanced-k2-system")["e1"]
    assert system["output"] == REPAIRED
    assert system["selected_round"] == 2
    followup = _rows(paths, "agent2-balanced-k2-followup")["e1"]
    assert followup["output"] == FIRST_DRIFTING
    assert followup["selected_round"] == 1
    restate = _rows(paths, "agent2-k0-restate")["e1"]
    assert RESTATE_LEAD not in str(restate["output"])
    assert restate["termination"] == "max_rounds"
    for key in keys:
        assert _rows(paths, key)["e2"]["output"] == FIRST_CLEAN
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["experiment"] == "v2"
    assert manifest["echo_guard"] is True
    assert manifest["primary_arm"] == V2.primary_arm == "agent2-balanced-k2-system"
    assert manifest["arms"]["agent2-balanced-k2-restate"]["echo_stripped_rounds"] >= 1
    assert manifest["arms"]["agent2-balanced-k2-system"]["echo_stripped_rounds"] == 0
    assert MARKER not in paths["manifest"].read_text(encoding="utf-8")

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
    summary = eval_main(
        [
            *("--experiment", "v2"),
            *("--generations-dir", str(paths["generations"])),
            *("--pairs", str(paths["pairs"])),
            *("--config", str(paths["config"])),
            *("--reports-dir", str(paths["reports"])),
            *("--cache-dir", str(tmp_path / "cache")),
        ],
        timestamp="2026-10-09T00:00:00Z",
    )
    assert (paths["reports"] / "agent2-eval.json").is_file()
    markdown = (paths["reports"] / "agent2-eval.md").read_text(encoding="utf-8")
    assert "SPEC 4.9" in markdown
    assert MARKER not in markdown
    primary = summary["primary"]
    blocks = summary["arms"]
    assert isinstance(primary, dict)
    assert isinstance(blocks, list)
    assert primary["arm"] == "agent2-balanced-k2-system"
    assert primary["vs_control"]["changed_cases"] == 1
    by_name = {block["name"]: block for block in blocks}
    assert len(by_name) == 6
    assert by_name["agent2-balanced-k2-system"]["role"] == "主臂"
    assert by_name["agent2-balanced-k2-followup"]["role"] == "探索性"
    # Nothing that survives the guard still carries the feedback.
    assert all(block["feedback_echo"]["final_cases"] == 0 for block in blocks)
    assert by_name["agent2-k0-followup"]["echo_stripped_rounds"] >= 1
