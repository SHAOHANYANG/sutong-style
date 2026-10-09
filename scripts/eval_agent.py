"""Score the agent files offline and write the pre-registered comparison (SPEC 4.8 / 4.9).

--skip-judge is the only path: this module never builds a judge client and never
reads .env. The verifier inside the loop and the three fidelity metrics here come
from the same extraction, so the summary always reports the metrics the loop did
not optimize next to the ones it did.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import structlog

from agent.prompts import LEADS
from eval.run_eval import evaluate, load_config, load_style_reference, output_hash
from scripts.eval_sweep import (
    TARGETS,
    _aggregate,
    _case_order,
    _metric_map,
    _num,
    _paired,
    judge_difference,
    mean_interval,
    status_targets,
)
from scripts.run_agent_eval import (
    EXPERIMENTS,
    TERMINATIONS,
    V1,
    Arm,
    Experiment,
    arms,
    echo_stripped_rounds,
    first_round_violation_cases,
    round_distribution,
    violation_transitions,
)
from scripts.train import Pair, load_pairs
from stylometry.distance import StyleReference
from stylometry.features import FEATURE_NAMES

LOGGER = structlog.get_logger()
CONTROL_KEY = "retrieval-balanced-k2"
CUMULATIVE_KEY = "retrieval-k0"
NUMERAL_FEATURE = "cjk_numeral_ratio"
# Metrics the loop selects on. Their improvement is partly true by construction.
OPTIMIZED = (
    ("entity_recall", "higher"),
    ("numeral_recall", "higher"),
    ("hallucination_rate", "lower"),
)
# Not selected on. profile_gap_per_case keeps the direction of SPEC 4.7.
UNOPTIMIZED_JUDGED = (("profile_gap_per_case", "lower"),)
UNOPTIMIZED_REPORT_ONLY = ("copy_ratio", "style_distance", "length_ratio", "cjk_numeral_z")
AGGREGATE_NAMES = (
    "style_distance",
    "profile_gap",
    "profile_gap_per_case",
    "entity_recall",
    "numeral_recall",
    "hallucination_rate",
    "copy_ratio",
)
SAME_RULES_NOTE = (
    "校验器与评估用的是同一套规则：agent 按「违规最少」挑选最终版本，"
    "而违规就是从计算 entity_recall、numeral_recall、hallucination_rate 的同一份抽取结果里导出的。"
    "这三项的改善在一定程度上是构造使然，不能单独作为事实漂移被治好的证据；"
    "规则抓不到的语义漂移 agent 同样看不见。必须连同「没有优化的」那张表一起读。"
)


def compare(
    agent_report: Mapping[str, object],
    baseline_report: Mapping[str, object],
    order: Sequence[str],
) -> dict[str, object]:
    """Paired differences (agent minus baseline) with the SPEC 4.7 interval and wording."""
    optimized: dict[str, object] = {}
    for name, direction in OPTIMIZED:
        optimized[name] = _judged(agent_report, baseline_report, order, name, direction)
    unoptimized: dict[str, object] = {}
    for name, direction in UNOPTIMIZED_JUDGED:
        unoptimized[name] = _judged(agent_report, baseline_report, order, name, direction)
    for name in UNOPTIMIZED_REPORT_ONLY:
        mean, low, high = mean_interval(_paired(agent_report, baseline_report, order, name))
        unoptimized[name] = {"conclusion": None, "high": high, "low": low, "mean_diff": mean}
    agent_gap = _number(_aggregate(agent_report)["profile_gap"])
    baseline_gap = _number(_aggregate(baseline_report)["profile_gap"])
    unoptimized["profile_gap"] = {
        "agent": agent_gap,
        "baseline": baseline_gap,
        "conclusion": None,
        "diff": agent_gap - baseline_gap,
    }
    unoptimized["cjk_numeral_ratio_profile_mean_delta"] = {
        "agent": _numeral_delta(agent_report),
        "baseline": _numeral_delta(baseline_report),
        "conclusion": None,
    }
    agent_hashes = _hashes(agent_report)
    baseline_hashes = _hashes(baseline_report)
    return {
        "changed_cases": sum(
            1 for doc_id in order if agent_hashes[doc_id] != baseline_hashes[doc_id]
        ),
        "n": len(order),
        "optimized": optimized,
        "unoptimized": unoptimized,
    }


def feedback_echo(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Post hoc diagnostic, not part of SPEC 4.8: outputs that repeat the revision feedback.

    The feedback quotes the missing fragment, so an output that copies the feedback
    back contains that fragment and passes the verifier without fixing the prose. A
    round counts as an echo when it holds a feedback lead verbatim, or a fragment of an
    earlier round's violation inside the same quote marks the feedback uses.
    """
    final_ids: list[str] = []
    second_round = 0
    second_round_total = 0
    revision_rounds = 0
    revision_rounds_total = 0
    for row in rows:
        rounds = [_obj(item) for item in _list(row.get("rounds"))]
        fragments: set[str] = set()
        for index, record in enumerate(rounds):
            if index > 0:
                text = str(record["output"])
                echoed = any(lead in text for lead in LEADS) or any(
                    f"「{fragment}」" in text for fragment in fragments
                )
                revision_rounds += int(echoed)
                revision_rounds_total += 1
                if index == 1:
                    second_round += int(echoed)
                    second_round_total += 1
                if echoed and row.get("selected_round") == index + 1:
                    final_ids.append(str(row["id"]))
            for item in _list(record["violations"]):
                violation = _obj(item)
                fragments.update(
                    str(violation[name]) for name in ("expected", "actual") if violation.get(name)
                )
    return {
        "final_cases": len(final_ids),
        "final_ids": final_ids,
        "revision_rounds": revision_rounds,
        "revision_rounds_total": revision_rounds_total,
        "second_round": second_round,
        "second_round_total": second_round_total,
    }


def exit_condition(comparison: Mapping[str, object]) -> dict[str, object]:
    """PLAN Phase 2 exit: hallucination_rate clearly lower. Reported, never adjusted."""
    optimized = _obj(comparison["optimized"])
    conclusion = _obj(optimized["hallucination_rate"])["conclusion"]
    return {
        "hallucination_rate_conclusion": conclusion,
        "met": conclusion == "改善",
        "rule": "主臂对 retrieval-balanced-k2 的 hallucination_rate 区间不含 0 且方向为下降",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评估 agent 各臂并写汇总")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="v1")
    parser.add_argument("--generations-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("eval/.judge_cache"))
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None, *, timestamp: str | None = None) -> dict[str, object]:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    experiment = EXPERIMENTS[args.experiment]
    summary_path = args.summary or args.reports_dir / f"{experiment.prefix}-eval.json"
    markdown_path = args.markdown or args.reports_dir / f"{experiment.prefix}-eval.md"
    config = load_config(args.config).model_copy(update={"pipeline": "agent"})
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    rows: dict[str, list[dict[str, object]]] = {}
    for arm in arms(experiment):
        path = args.generations_dir / f"{arm.key}.jsonl"
        if not path.is_file():
            raise SystemExit(f"生成文件不存在: {path}")
        body = evaluate(
            config,
            path,
            arm.key,
            skip_judge=True,
            report_path=args.reports_dir / f"{arm.key}.json",
            cache_dir=args.cache_dir,
            completer=None,
        )
        reports[arm.key] = body.model_dump()
        rows[arm.key] = _read_rows(path)
    for key in (CONTROL_KEY, CUMULATIVE_KEY):
        reports[key] = _baseline_report(args.reports_dir / f"{key}.json")
        rows[key] = _read_rows(args.generations_dir / f"{key}.jsonl")
    summary = assemble_summary(
        reports=reports,
        rows=rows,
        pairs=load_pairs(args.pairs),
        reference=load_style_reference(config),
        timestamp=timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        experiment=experiment,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    markdown_path.write_text(str(summary["changelog_markdown"]), encoding="utf-8")
    LOGGER.info("agent_eval_written", summary=str(summary_path), arms=len(arms(experiment)))
    return summary


def assemble_summary(
    *,
    reports: Mapping[str, Mapping[str, object]],
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    pairs: Sequence[Pair],
    reference: StyleReference,
    timestamp: str,
    experiment: Experiment = V1,
) -> dict[str, object]:
    """Every arm, the primary paired intervals in two tables, rounds and violation moves."""
    by_id = {row.id: row for row in pairs}
    order = _case_order(reports[CONTROL_KEY])
    primary_arm = experiment.primary_arm
    scored: dict[str, dict[str, object]] = {}
    for key in (*(arm.key for arm in arms(experiment)), CONTROL_KEY, CUMULATIVE_KEY):
        _check_outputs(key, reports[key], rows[key])
        if _case_order(reports[key]) != order:
            raise SystemExit(f"样本顺序与对照不一致: {key}")
        scored[key] = _with_extras(reports[key], rows[key], by_id, reference)
    arm_blocks: list[dict[str, object]] = []
    exploratory: dict[str, object] = {}
    for arm in arms(experiment):
        _check_first_round(arm, rows[arm.key], rows[arm.baseline_key])
        comparison = compare(scored[arm.key], scored[arm.baseline_key], order)
        block = _arm_block(arm, scored[arm.key], rows[arm.key], comparison)
        block["role"] = "主臂" if arm.key == primary_arm else "探索性"
        arm_blocks.append(block)
        if arm.key != primary_arm:
            exploratory[arm.key] = {"baseline": arm.baseline_key, **comparison}
    vs_control = compare(scored[primary_arm], scored[CONTROL_KEY], order)
    summary: dict[str, object] = {
        "arms": arm_blocks,
        "baselines": [
            {"aggregate": _aggregate_block(scored[key]), "name": key}
            for key in (CONTROL_KEY, CUMULATIVE_KEY)
        ],
        "changelog_markdown": "",
        "exit_condition": exit_condition(vs_control),
        "experiment": experiment.name,
        "exploratory": exploratory,
        "human_spot_check": "pending",
        "primary": {
            "arm": primary_arm,
            "vs_control": {"baseline": CONTROL_KEY, **vs_control},
            "vs_k0": {
                "baseline": CUMULATIVE_KEY,
                **compare(scored[primary_arm], scored[CUMULATIVE_KEY], order),
            },
        },
        "same_rules_note": SAME_RULES_NOTE,
        "spec": experiment.spec,
        "targets": status_targets(_aggregate(reports[primary_arm])),
        "timestamp": timestamp,
    }
    summary["changelog_markdown"] = render_markdown(summary)
    return summary


def render_markdown(summary: Mapping[str, object]) -> str:
    """CHANGELOG tables. Exploratory arms are labeled and never replace the README row."""
    primary = _obj(summary["primary"])
    blocks = [_obj(item) for item in _list(summary["arms"])]
    lines = [
        f"主臂是 {primary['arm']}，对照是 {CONTROL_KEY}（同样 59 条，逐条配对，agent − 对照）。"
        f"规则见 SPEC {summary['spec']}。",
        "其余三臂是探索性的，不替换 README 的 agent 行。style_win_rate 本轮不跑。",
        "",
        SAME_RULES_NOTE,
        "",
        f"人工抽查：{summary['human_spot_check']}（由仓库所有者进行，不得由模型代填）。",
    ]
    for title, name in (("主臂 − 对照", "vs_control"), ("主臂 − k0（累计）", "vs_k0")):
        comparison = _obj(primary[name])
        lines.extend(["", f"**{title}（{comparison['baseline']}）：agent 直接优化的指标**", ""])
        lines.extend(_metric_table(_obj(comparison["optimized"])))
        lines.extend(["", f"**{title}（{comparison['baseline']}）：agent 没有优化的指标**", ""])
        lines.extend(_metric_table(_obj(comparison["unoptimized"])))
    lines.extend(
        [
            "",
            "| 组合 | 角色 | style_distance | profile_gap | profile_gap_per_case "
            "| entity_recall | numeral_recall | hallucination_rate | copy_ratio "
            "| 字数比 | cjk_numeral_ratio 逐维均值差 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for block in (*blocks, *(_obj(item) for item in _list(summary["baselines"]))):
        aggregate = _obj(block["aggregate"])
        cells = [str(block["name"]), str(block.get("role", "对照"))]
        cells.extend(_num(aggregate[name]) for name in AGGREGATE_NAMES)
        cells.extend([_num(aggregate["length_ratio"]), _num(aggregate["cjk_numeral_ratio_delta"])])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| 臂 | 1 轮 | 2 轮 | 3 轮 | accepted | max_rounds | recursion_limit | fallback "
            "| 选中第 1 轮 | 选中第 2 轮 | 选中第 3 轮 | 未选中 "
            "| 第一轮有违规的样本 | 最终输出不同于基线的样本 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for block in blocks:
        rounds = _obj(block["rounds"])
        total = _obj(rounds["total_rounds"])
        picked = _obj(rounds["selected_round"])
        ended = _obj(rounds["termination"])
        cells = [str(block["name"])]
        cells.extend(str(total.get(key, 0)) for key in ("1", "2", "3"))
        cells.extend(str(ended.get(key, 0)) for key in TERMINATIONS)
        cells.extend(str(picked.get(key, 0)) for key in ("1", "2", "3", "none"))
        cells.extend([str(block["first_round_violation_cases"]), str(block["changed_cases"])])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| 臂 | 违规类型 | 第一轮条数 | 最终条数 | 修掉 | 新引入 "
            "| 第一轮涉及样本 | 最终涉及样本 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for block in blocks:
        for kind, cell in _obj(block["violation_transitions"]).items():
            item = _obj(cell)
            cells = [str(block["name"]), kind]
            cells.extend(
                str(item[name])
                for name in ("first", "final", "fixed", "introduced", "first_cases", "final_cases")
            )
            lines.append("| " + " | ".join(cells) + " |")
    fallbacks = [
        f"{block['name']}：{_obj(item)['id']}（{_obj(item)['fallback_kind']}）"
        for block in blocks
        for item in _list(block["fallbacks"])
    ]
    lines.extend(["", "兜底样本：" + ("；".join(fallbacks) if fallbacks else "无") + "。"])
    lines.extend(
        [
            "",
            "**反馈回显**（SPEC 4.8 的事后诊断，不在那一轮的预注册里；4.9 起是预注册的报告项）",
            "",
            "修订反馈里用「」引用了缺失的片段。输出若把反馈抄回去，片段就出现在输出里，"
            "校验器判为已修复。下表数的是仍含反馈引导语或「片段」的输出；"
            "「被防护切过的轮」只在打开回显防护时非零，切掉之后的文本不再计入前几列。",
            "",
            "| 臂 | 最终输出改动的样本 | 其中最终输出含反馈回显 | 修订轮含回显 "
            "| 第 2 轮含回显 | 被防护切过的轮 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for block in blocks:
        echo = _obj(block["feedback_echo"])
        picked = _obj(_obj(block["rounds"])["selected_round"])
        revised = sum(_count(picked.get(key, 0)) for key in ("2", "3"))
        lines.append(
            f"| {block['name']} | {revised} | {echo['final_cases']} "
            f"| {echo['revision_rounds']} / {echo['revision_rounds_total']} "
            f"| {echo['second_round']} / {echo['second_round_total']} "
            f"| {block['echo_stripped_rounds']} |"
        )
    lines.extend(["", "**探索臂（各自对第一轮所取的文件，未做多重比较校正）**", ""])
    lines.extend(["| 臂 | 基线 | 指标 | 均值差 | 95% 区间 | 结论 |", "|---|---|---|---|---|---|"])
    for key, value in _obj(summary["exploratory"]).items():
        comparison = _obj(value)
        for group in ("optimized", "unoptimized"):
            for name, cell in _obj(comparison[group]).items():
                item = _obj(cell)
                if "mean_diff" not in item:
                    continue
                conclusion = item["conclusion"] if item["conclusion"] is not None else "只报告"
                interval = f"[{_num(item['low'])}, {_num(item['high'])}]"
                lines.append(
                    f"| {key} | {comparison['baseline']} | {name} "
                    f"| {_num(item['mean_diff'])} | {interval} | {conclusion} |"
                )
    targets = _obj(summary["targets"])
    lines.extend(["", "| 目标 | 阈值 | 主臂 | 达到 |", "|---|---|---|---|"])
    for name, _op, _threshold in TARGETS:
        item = _obj(targets[name])
        reached = "达到" if item["met"] else "未达到"
        lines.append(
            f"| {name} | {item['op']} {item['threshold']} | {_num(item['value'])} | {reached} |"
        )
    condition = _obj(summary["exit_condition"])
    reached = "达到" if condition["met"] else "未达到"
    lines.extend(
        [
            "",
            f"PLAN 出口条件（hallucination_rate 明显下降）：{reached}"
            f"（结论：{condition['hallucination_rate_conclusion']}）。",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_table(metrics: Mapping[str, object]) -> list[str]:
    lines = ["| 指标 | 均值差 | 95% 区间 | 结论 |", "|---|---|---|---|"]
    for name, cell in metrics.items():
        item = _obj(cell)
        if "mean_diff" in item:
            conclusion = item["conclusion"] if item["conclusion"] is not None else "只报告"
            interval = f"[{_num(item['low'])}, {_num(item['high'])}]"
            lines.append(f"| {name} | {_num(item['mean_diff'])} | {interval} | {conclusion} |")
        elif "diff" in item:
            lines.append(f"| {name} | {_num(item['diff'])} | — | 只报点估计 |")
        else:
            both = f"agent {_num(item['agent'])}，基线 {_num(item['baseline'])}"
            lines.append(f"| {name} | {both} | — | 只报告 |")
    return lines


def _judged(
    agent_report: Mapping[str, object],
    baseline_report: Mapping[str, object],
    order: Sequence[str],
    name: str,
    direction: str,
) -> dict[str, object]:
    mean, low, high = mean_interval(_paired(agent_report, baseline_report, order, name))
    return {
        "conclusion": judge_difference(low, high, direction),
        "high": high,
        "low": low,
        "mean_diff": mean,
    }


def _arm_block(
    arm: Arm,
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    comparison: Mapping[str, object],
) -> dict[str, object]:
    return {
        "aggregate": _aggregate_block(report),
        "baseline": arm.baseline_key,
        "changed_cases": comparison["changed_cases"],
        "echo_stripped_rounds": echo_stripped_rounds(rows),
        "exemplar_source": arm.source,
        "fallbacks": [
            {"fallback_kind": row.get("fallback_kind"), "id": row["id"]}
            for row in rows
            if row["termination"] == "fallback"
        ],
        "feedback_echo": feedback_echo(rows),
        "feedback_format": arm.feedback_format,
        "first_round_violation_cases": first_round_violation_cases(rows),
        "name": arm.key,
        "rounds": round_distribution(rows),
        "violation_transitions": violation_transitions(rows),
    }


def _aggregate_block(report: Mapping[str, object]) -> dict[str, object]:
    aggregate = _aggregate(report)
    block: dict[str, object] = {name: aggregate.get(name) for name in AGGREGATE_NAMES}
    ratios = _metric_map(report, "length_ratio")
    block["length_ratio"] = sum(ratios.values()) / len(ratios)
    block["cjk_numeral_ratio_delta"] = _numeral_delta(report)
    return block


def _with_extras(
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    by_id: Mapping[str, Pair],
    reference: StyleReference,
) -> dict[str, object]:
    """Copy of the report whose per-case metrics also hold the two SPEC 4.8 extras."""
    outputs = {str(row["id"]): str(row["output"]) for row in rows}
    index = FEATURE_NAMES.index(NUMERAL_FEATURE)
    per_case: list[dict[str, object]] = []
    for item in _list(report["per_case"]):
        case = _obj(item)
        doc_id = str(case["id"])
        pair = by_id.get(doc_id)
        if pair is None or not pair.original:
            raise SystemExit(f"样本没有原文，无法算字数比: {doc_id}")
        output = outputs[doc_id]
        metrics = {
            **_obj(case["metrics"]),
            "cjk_numeral_z": float(reference.zscore(output)[index]),
            "length_ratio": len(output) / len(pair.original),
        }
        per_case.append({**case, "metrics": metrics})
    return {**report, "per_case": per_case}


def _check_outputs(
    key: str,
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """A report scored from another generation file must not be compared."""
    hashes = _hashes(report)
    outputs = {str(row["id"]): str(row["output"]) for row in rows}
    for doc_id, digest in hashes.items():
        if doc_id not in outputs or output_hash(outputs[doc_id]) != digest:
            raise SystemExit(f"报告与生成文件对不上: {key} {doc_id}")


def _check_first_round(
    arm: Arm,
    rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> None:
    baseline = {str(row["id"]): row["output"] for row in baseline_rows}
    for row in rows:
        rounds = _list(row.get("rounds"))
        if not rounds or _obj(rounds[0])["output"] != baseline.get(str(row["id"])):
            raise SystemExit(f"第一轮输出与基线不同: {row.get('id')} {arm.key}")


def _baseline_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"基线报告不存在，先跑 scripts.eval_sweep: {path}")
    return _obj(json.loads(path.read_text(encoding="utf-8")))


def _hashes(report: Mapping[str, object]) -> dict[str, str]:
    return {
        str(_obj(item)["id"]): str(_obj(item)["output_hash"]) for item in _list(report["per_case"])
    }


def _numeral_delta(report: Mapping[str, object]) -> float:
    delta = _obj(report.get("profile_mean_delta"))
    return _number(delta[NUMERAL_FEATURE])


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"生成文件不存在: {path}")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_obj(json.loads(line)))
    return rows


def _obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SystemExit("汇总结构损坏：不是对象")
    return {str(key): item for key, item in value.items()}


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise SystemExit("汇总结构损坏：不是列表")
    return list(value)


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit("汇总结构损坏：不是整数")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit("汇总结构损坏：不是数")
    return float(value)


if __name__ == "__main__":
    main()
