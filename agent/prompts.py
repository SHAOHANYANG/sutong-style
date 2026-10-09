"""Revision feedback messages. Surfaces come from Violation; never paraphrase vaguely."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from agent.state import AgentState
from eval.fidelity import Violation, ViolationKind
from retrieval.prompt import Exemplar, build_prompt

FeedbackFormat = Literal["followup", "restate", "system"]

KIND_ORDER: tuple[ViolationKind, ...] = (
    "entity_missing",
    "numeral_missing",
    "entity_hallucination",
    "title",
)

FOLLOWUP_LEAD = "上一版存在以下问题，请按条修正后重写，其余部分尽量保留："
RESTATE_LEAD = "请按下列要求改写这段输入，其余内容尽量保留："
SYSTEM_LEAD = "改写时必须满足下列要求："
LEADS = (FOLLOWUP_LEAD, RESTATE_LEAD, SYSTEM_LEAD)


def _quote(text: str) -> str:
    return f"「{text}」"


def format_violation(item: Violation) -> str:
    """One concrete sentence. Fragment text is taken from the Violation fields."""
    if item.kind == "entity_missing":
        if not item.expected:
            raise ValueError("entity_missing 缺少 expected")
        return f"输入里的{_quote(item.expected)}在上一版里不见了，请写回去。"
    if item.kind == "numeral_missing":
        if not item.expected:
            raise ValueError("numeral_missing 缺少 expected")
        return (
            f"输入里的数值{_quote(item.expected)}必须保留且数值不变，"
            f"请用中文数字书写，不要改成阿拉伯数字。"
        )
    if item.kind == "entity_hallucination":
        if not item.actual:
            raise ValueError("entity_hallucination 缺少 actual")
        return f"上一版里多出了输入中没有的{_quote(item.actual)}，请去掉。"
    if item.kind == "title":
        if not item.expected:
            raise ValueError("title 缺少 expected")
        if item.actual:
            return (
                f"输入里的称谓{_quote(item.expected)}被改成了{_quote(item.actual)}，"
                f"请改回{_quote(item.expected)}。"
            )
        return f"输入里的称谓{_quote(item.expected)}在上一版里不见了，请写回去。"
    raise ValueError(f"未知违规类型: {item.kind}")


def _sort_key(item: Violation) -> tuple[int, str, str]:
    try:
        kind_rank = KIND_ORDER.index(item.kind)
    except ValueError as exc:
        raise ValueError(f"未知违规类型: {item.kind}") from exc
    return (kind_rank, item.expected or "", item.actual or "")


def _dedupe_key(item: Violation) -> tuple[str, str | None, str | None]:
    return (item.kind, item.expected, item.actual)


def ordered_violations(violations: Sequence[Violation]) -> list[Violation]:
    """Stable unique order: kind, then expected, then actual."""
    unique: list[Violation] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for item in sorted(violations, key=_sort_key):
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def feedback_lines(violations: Sequence[Violation]) -> list[str]:
    return [format_violation(item) for item in ordered_violations(violations)]


def feedback_block(violations: Sequence[Violation], *, lead: str) -> str:
    lines = feedback_lines(violations)
    if not lines:
        raise ValueError("反馈块不能没有违规")
    return lead + "\n" + "\n".join(f"- {line}" for line in lines)


def build_revision_messages(
    vernacular: str,
    exemplars: Sequence[Exemplar],
    *,
    previous_output: str,
    violations: Sequence[Violation],
    feedback_format: FeedbackFormat = "followup",
) -> list[dict[str, str]]:
    """Build the next-round chat. Round-1 shape is always build_prompt."""
    base = build_prompt(vernacular, exemplars)
    if feedback_format == "followup":
        return [
            *base,
            {"role": "assistant", "content": previous_output},
            {
                "role": "user",
                "content": feedback_block(violations, lead=FOLLOWUP_LEAD),
            },
        ]
    if feedback_format == "restate":
        body = vernacular + "\n\n" + feedback_block(violations, lead=RESTATE_LEAD)
        return [*base[:-1], {"role": "user", "content": body}]
    if feedback_format == "system":
        # The model treats every user turn as text to rewrite, so the instruction
        # goes where training put the only instruction it ever saw.
        block = feedback_block(violations, lead=SYSTEM_LEAD)
        system = {"role": "system", "content": base[0]["content"] + "\n\n" + block}
        return [system, *base[1:]]
    raise ValueError(f"未知反馈格式: {feedback_format}")


def strip_feedback_echo(text: str, violations: Sequence[Violation]) -> tuple[str, int]:
    """Cut feedback the model copied back. Returns the kept text and the characters removed.

    The feedback quotes each missing fragment, so a copied instruction makes the
    fragment appear in the output and the verifier pass. The cut starts at the first
    verbatim lead, or at the line holding a quoted fragment of the feedback just sent
    (one line earlier when that line ends with a colon: a reworded lead).
    """
    cuts = [text.index(lead) for lead in LEADS if lead in text]
    fragments = {
        _quote(fragment)
        for item in violations
        for fragment in (item.expected, item.actual)
        if fragment
    }
    lines = text.split("\n")
    offset = 0
    previous_start = 0
    for index, line in enumerate(lines):
        if any(fragment in line for fragment in fragments):
            reworded_lead = index > 0 and lines[index - 1].rstrip().endswith(("：", ":"))
            cuts.append(previous_start if reworded_lead else offset)
            break
        previous_start = offset
        offset += len(line) + 1
    if not cuts:
        return text, 0
    kept = text[: min(cuts)].rstrip()
    return kept, len(text) - len(kept)


def make_revision_message_builder(
    feedback_format: FeedbackFormat = "followup",
) -> Callable[[AgentState], list[dict[str, str]]]:
    """MessageBuilder: revise when a previous output still has violations."""

    def build(state: AgentState) -> list[dict[str, str]]:
        vernacular = state["input"]
        exemplars = list(state.get("exemplars") or [])
        candidates = list(state.get("candidates") or [])
        violations = list(state.get("last_violations") or [])
        if not candidates or not violations:
            return build_prompt(vernacular, exemplars)
        return build_revision_messages(
            vernacular,
            exemplars,
            previous_output=candidates[-1],
            violations=violations,
            feedback_format=feedback_format,
        )

    return build
