"""Score the thirteen sweep files offline and write the pre-registered comparison.

--skip-judge is the only path: this module never builds a judge client and never
reads .env. Conclusions follow SPEC 4.7. Exploratory groups stay in the table.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog

from eval.exemplar_diagnostics import exemplar_copy_ratio, exemplar_entity_leak
from eval.run_eval import evaluate, load_config, load_gazetteer, summarize
from scripts.sweep_topk import FUSION_ORDER, group_key, jobs
from scripts.train import Pair, load_pairs

LOGGER = structlog.get_logger()
PRIMARY_CONFIG = "balanced"
PRIMARY_K = 2
N_BOOT = 10_000
BOOT_SEED = 42
LOWER_IS_BETTER = ("profile_gap_per_case", "hallucination_rate")
HIGHER_IS_BETTER = ("entity_recall", "numeral_recall")
REPORT_ONLY = ("copy_ratio", "style_distance")
DIAGNOSTICS = ("exemplar_entity_leak", "exemplar_copy_ratio")
TARGETS = (
    ("numeral_recall", ">=", 0.92),
    ("hallucination_rate", "<=", 0.005),
    ("profile_gap", "<", 0.204),
)


def mean_interval(
    diffs: Sequence[float],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Mean of the paired differences and the 2.5/97.5 percentiles of its bootstrap."""
    array = np.asarray(list(diffs), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise SystemExit("配对差为空或含非有限值")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def judge_difference(low: float, high: float, direction: str) -> str:
    """SPEC 4.7: an interval that contains 0 is not a difference, in either direction."""
    if low <= 0.0 <= high:
        return "未检出差异"
    if direction == "higher":
        return "改善" if low > 0.0 else "恶化"
    if direction == "lower":
        return "改善" if high < 0.0 else "恶化"
    raise ValueError(f"未知方向: {direction}")


def status_targets(aggregate: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Whether the three STATUS 9.5 numbers were met. Not a decision rule."""
    report: dict[str, dict[str, object]] = {}
    for name, op, threshold in TARGETS:
        value = aggregate.get(name)
        if not isinstance(value, (int, float)):
            raise SystemExit(f"主配置缺少指标: {name}")
        number = float(value)
        if op == ">=":
            met = number >= threshold
        elif op == "<=":
            met = number <= threshold
        else:
            met = number < threshold
        report[name] = {"met": met, "op": op, "threshold": threshold, "value": number}
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评估十三组检索生成并写汇总")
    parser.add_argument("--generations-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--summary", type=Path, default=Path("eval/reports/retrieval-sweep.json"))
    parser.add_argument("--markdown", type=Path, default=Path("eval/reports/retrieval-sweep.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("eval/.judge_cache"))
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None, *, timestamp: str | None = None) -> dict[str, object]:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    config = load_config(args.config).model_copy(update={"pipeline": "retrieval"})
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    for config_name, k in jobs():
        key = group_key(config_name, k)
        path = args.generations_dir / f"{key}.jsonl"
        if not path.is_file():
            raise SystemExit(f"生成文件不存在: {path}")
        body = evaluate(
            config,
            path,
            key,
            skip_judge=True,
            report_path=args.reports_dir / f"{key}.json",
            cache_dir=args.cache_dir,
            completer=None,
        )
        reports[key] = body.model_dump()
    summary = assemble_summary(
        reports=reports,
        generations_dir=args.generations_dir,
        pairs=load_pairs(args.pairs),
        gazetteer=load_gazetteer(Path(config.gazetteer)),
        timestamp=timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.summary.write_text(text, encoding="utf-8")
    args.markdown.write_text(str(summary["changelog_markdown"]), encoding="utf-8")
    LOGGER.info("sweep_eval_written", summary=str(args.summary), groups=len(reports))
    return summary


def assemble_summary(
    *,
    reports: Mapping[str, Mapping[str, object]],
    generations_dir: Path,
    pairs: Sequence[Pair],
    gazetteer: set[str],
    timestamp: str,
) -> dict[str, object]:
    """One table, the primary paired intervals, and the diagnostic natural level."""
    by_id = {row.id: row for row in pairs}
    primary_key = group_key(PRIMARY_CONFIG, PRIMARY_K)
    control_key = group_key(None, 0)
    rows = {
        group_key(config, k): _read_rows(generations_dir / f"{group_key(config, k)}.jsonl")
        for config, k in jobs()
    }
    primary_rows = rows[primary_key]
    for row in primary_rows:
        if len(_ids(row)) != PRIMARY_K:
            raise SystemExit(f"主配置范例不是 {PRIMARY_K} 条: {row.get('id')}")
    diagnostics = {
        key: _diagnostics(
            rows[key],
            rows[primary_key],
            by_id,
            gazetteer,
            natural=key == control_key,
        )
        for key in rows
    }
    order = _case_order(reports[control_key])
    primary = _primary_block(reports[primary_key], reports[control_key], diagnostics, order)
    groups = [_group_block(key, reports[key], diagnostics[key]) for key in _group_order()]
    summary: dict[str, object] = {
        "changelog_markdown": "",
        "diagnostics": {key: _diagnostic_block(diagnostics[key]) for key in _group_order()},
        "groups": groups,
        "primary": primary,
        "targets": status_targets(_aggregate(reports[primary_key])),
        "timestamp": timestamp,
    }
    summary["changelog_markdown"] = render_markdown(summary)
    return summary


def render_markdown(summary: Mapping[str, object]) -> str:
    """A CHANGELOG table. Exploratory rows are labeled and do not replace the main row."""
    primary = summary["primary"]
    groups = summary["groups"]
    targets = summary["targets"]
    if not isinstance(primary, dict):
        raise SystemExit("汇总结构损坏")
    if not isinstance(groups, list):
        raise SystemExit("汇总结构损坏")
    if not isinstance(targets, dict):
        raise SystemExit("汇总结构损坏")
    lines = [
        "主配置是 balanced、k = 2，对照是同一次扫描的 k = 0。",
        "其余组合是探索性的，不替换 README 的 retrieval 行。style_win_rate 本轮不跑。",
        "",
        "| 指标 | 均值差（主 − 对照） | 95% 区间 | 结论 |",
        "|---|---|---|---|",
    ]
    metrics = primary["metrics"]
    if not isinstance(metrics, dict):
        raise SystemExit("主比较缺少指标")
    for name in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER, *REPORT_ONLY, "profile_gap"):
        item = metrics[name]
        if not isinstance(item, dict):
            raise SystemExit(f"指标不是对象: {name}")
        if name == "profile_gap":
            lines.append(f"| {name} | {_num(item['diff'])} | — | 只报点估计 |")
            continue
        conclusion = item["conclusion"] if item["conclusion"] is not None else "只报告"
        mean = _num(item["mean_diff"])
        interval = f"[{_num(item['low'])}, {_num(item['high'])}]"
        lines.append(f"| {name} | {mean} | {interval} | {conclusion} |")
    header = (
        "| 组合 | 角色 | style_distance | profile_gap | profile_gap_per_case "
        "| entity_recall | numeral_recall | hallucination_rate | copy_ratio |"
    )
    rule = "|---|---|---|---|---|---|---|---|---|"
    lines.extend(["", header, rule])
    for group in groups:
        if not isinstance(group, dict):
            raise SystemExit("分组不是对象")
        aggregate = group["aggregate"]
        if not isinstance(aggregate, dict):
            raise SystemExit("聚合不是对象")
        cells = [
            str(group["name"]),
            str(group["role"]),
            _num(aggregate["style_distance"]),
            _num(aggregate["profile_gap"]),
            _num(aggregate["profile_gap_per_case"]),
            _num(aggregate["entity_recall"]),
            _num(aggregate["numeral_recall"]),
            _num(aggregate["hallucination_rate"]),
            _num(aggregate["copy_ratio"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "| 目标 | 阈值 | 主配置 | 达到 |", "|---|---|---|---|"])
    for name, _op, _threshold in TARGETS:
        item = targets[name]
        if not isinstance(item, dict):
            raise SystemExit(f"目标不是对象: {name}")
        reached = "达到" if item["met"] else "未达到"
        lines.append(
            f"| {name} | {item['op']} {item['threshold']} | {_num(item['value'])} | {reached} |"
        )
    lines.append("")
    return "\n".join(lines)


def _primary_block(
    main_report: Mapping[str, object],
    control_report: Mapping[str, object],
    diagnostics: Mapping[str, dict[str, dict[str, float]]],
    order: Sequence[str],
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, direction in (
        *((item, "lower") for item in LOWER_IS_BETTER),
        *((item, "higher") for item in HIGHER_IS_BETTER),
    ):
        mean, low, high = mean_interval(_paired(main_report, control_report, order, name))
        metrics[name] = {
            "conclusion": judge_difference(low, high, direction),
            "high": high,
            "low": low,
            "mean_diff": mean,
        }
    for name in REPORT_ONLY:
        mean, low, high = mean_interval(_paired(main_report, control_report, order, name))
        metrics[name] = {"conclusion": None, "high": high, "low": low, "mean_diff": mean}
    main_gap = _aggregate(main_report)["profile_gap"]
    control_gap = _aggregate(control_report)["profile_gap"]
    if not isinstance(main_gap, (int, float)) or not isinstance(control_gap, (int, float)):
        raise SystemExit("profile_gap 不是数")
    metrics["profile_gap"] = {
        "conclusion": None,
        "control": float(control_gap),
        "diff": float(main_gap) - float(control_gap),
        "main": float(main_gap),
    }
    primary_key = group_key(PRIMARY_CONFIG, PRIMARY_K)
    control_key = group_key(None, 0)
    diagnostic_report: dict[str, object] = {}
    for name in DIAGNOSTICS:
        diffs = [
            diagnostics[primary_key][doc_id][name] - diagnostics[control_key][doc_id][name]
            for doc_id in order
        ]
        mean, low, high = mean_interval(diffs)
        control_values = [diagnostics[control_key][doc_id][name] for doc_id in order]
        main_values = [diagnostics[primary_key][doc_id][name] for doc_id in order]
        diagnostic_report[name] = {
            "control_mean": float(np.mean(control_values)),
            "high": high,
            "low": low,
            "main_mean": float(np.mean(main_values)),
            "mean_diff": mean,
        }
    return {
        "config": PRIMARY_CONFIG,
        "control": control_key,
        "diagnostics": diagnostic_report,
        "k": PRIMARY_K,
        "metrics": metrics,
    }


def _group_block(
    key: str,
    report: Mapping[str, object],
    diagnostic: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    config, k = _split_key(key)
    if key == group_key(None, 0):
        role = "对照"
    elif key == group_key(PRIMARY_CONFIG, PRIMARY_K):
        role = "主配置"
    else:
        role = "探索性"
    return {
        "aggregate": {
            name: _aggregate(report).get(name)
            for name in (
                "style_distance",
                "profile_gap",
                "profile_gap_per_case",
                "entity_recall",
                "numeral_recall",
                "hallucination_rate",
                "copy_ratio",
                "style_win_rate",
            )
        },
        "diagnostics": _diagnostic_block(diagnostic),
        "fusion_config": config,
        "k": k,
        "name": key,
        "role": role,
    }


def _diagnostic_block(diagnostic: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    block: dict[str, object] = {}
    for name in DIAGNOSTICS:
        values = [row[name] for row in diagnostic.values()]
        block[name] = summarize(values)
    return block


def _diagnostics(
    rows: Sequence[Mapping[str, object]],
    primary_rows: Sequence[Mapping[str, object]],
    by_id: Mapping[str, Pair],
    gazetteer: set[str],
    *,
    natural: bool,
) -> dict[str, dict[str, float]]:
    primary_ids = {str(row["id"]): _ids(row) for row in primary_rows}
    scored: dict[str, dict[str, float]] = {}
    for row in rows:
        doc_id = str(row["id"])
        exemplar_ids = primary_ids[doc_id] if natural else _ids(row)
        exemplars = [_pair_text(by_id, item) for item in exemplar_ids]
        output = str(row["output"])
        query = by_id.get(doc_id)
        if query is None:
            raise SystemExit(f"查询不在 pairs 里: {doc_id}")
        originals = [original for _text, original in exemplars]
        scored[doc_id] = {
            "exemplar_copy_ratio": exemplar_copy_ratio(output, originals),
            "exemplar_entity_leak": exemplar_entity_leak(
                query.vernacular, output, exemplars, gazetteer
            ),
        }
    return scored


def _paired(
    main_report: Mapping[str, object],
    control_report: Mapping[str, object],
    order: Sequence[str],
    name: str,
) -> list[float]:
    main_values = _metric_map(main_report, name)
    control_values = _metric_map(control_report, name)
    return [main_values[doc_id] - control_values[doc_id] for doc_id in order]


def _metric_map(report: Mapping[str, object], name: str) -> dict[str, float]:
    per_case = report.get("per_case")
    if not isinstance(per_case, list):
        raise SystemExit("报告缺少 per_case")
    values: dict[str, float] = {}
    for row in per_case:
        if not isinstance(row, dict):
            raise SystemExit("per_case 行不是对象")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or name not in metrics:
            raise SystemExit(f"缺少逐条指标: {name}")
        value = metrics[name]
        if not isinstance(value, (int, float)):
            raise SystemExit(f"逐条指标不是数: {name} {row.get('id')}")
        values[str(row["id"])] = float(value)
    return values


def _case_order(report: Mapping[str, object]) -> list[str]:
    per_case = report.get("per_case")
    if not isinstance(per_case, list):
        raise SystemExit("报告缺少 per_case")
    return [str(row["id"]) for row in per_case if isinstance(row, dict)]


def _aggregate(report: Mapping[str, object]) -> Mapping[str, object]:
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict):
        raise SystemExit("报告缺少 aggregate")
    return aggregate


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SystemExit(f"生成文件行不是对象: {path}")
        rows.append({str(key): value for key, value in payload.items()})
    return rows


def _ids(row: Mapping[str, object]) -> list[str]:
    value = row.get("exemplar_ids")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"exemplar_ids 损坏: {row.get('id')}")
    return [str(item) for item in value]


def _pair_text(by_id: Mapping[str, Pair], doc_id: str) -> tuple[str, str]:
    row = by_id.get(doc_id)
    if row is None:
        raise SystemExit(f"范例不在 pairs 里: {doc_id}")
    return row.vernacular, row.original


def _split_key(key: str) -> tuple[str | None, int]:
    if key == "retrieval-k0":
        return None, 0
    prefix = "retrieval-"
    config, raw_k = key[len(prefix) :].rsplit("-k", 1)
    if config not in FUSION_ORDER:
        raise SystemExit(f"未知分组: {key}")
    return config, int(raw_k)


def _group_order() -> list[str]:
    return [group_key(config, k) for config, k in jobs()]


def _num(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
