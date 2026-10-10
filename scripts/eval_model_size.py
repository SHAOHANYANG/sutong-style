"""Score a larger base against the 3B results and write the pre-registered comparison (SPEC 4.10).

Runs offline: the judge is skipped and nothing here reads .env. The generations
come from scripts/run_model_size.py on a rented GPU; the 3B files already exist.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import structlog

from eval.run_eval import evaluate, load_config, output_hash, summarize
from scripts.eval_sweep import (
    _aggregate,
    _case_order,
    _num,
    _paired,
    judge_difference,
    mean_interval,
)
from scripts.vernacularize import text_similarity

LOGGER = structlog.get_logger()
JUDGED = (
    ("profile_gap_per_case", "lower"),
    ("entity_recall", "higher"),
    ("numeral_recall", "higher"),
    ("hallucination_rate", "lower"),
)
REPORT_ONLY = ("copy_ratio", "style_distance")
SENTENCE_FEATURES = ("sent_len_std", "sent_len_p90")
AGGREGATE_NAMES = (
    "style_distance",
    "profile_gap",
    "profile_gap_per_case",
    "entity_recall",
    "numeral_recall",
    "hallucination_rate",
    "copy_ratio",
)
# At or above this similarity the output is the input with a few characters changed.
NEAR_COPY = 0.95
PROBE_NOTE = (
    "探针输入由 AI 助手撰写，不是人工语料，没有参照原文。"
    "这里只量输出与输入有多像，不说明改得好不好，不能替代 human_eval。"
)


def compare(
    new_report: Mapping[str, object],
    old_report: Mapping[str, object],
    order: Sequence[str],
) -> dict[str, object]:
    """Paired differences (new minus old) with the SPEC 4.7 interval and wording."""
    metrics: dict[str, object] = {}
    for name, direction in JUDGED:
        mean, low, high = mean_interval(_paired(new_report, old_report, order, name))
        metrics[name] = {
            "conclusion": judge_difference(low, high, direction),
            "high": high,
            "low": low,
            "mean_diff": mean,
        }
    for name in REPORT_ONLY:
        mean, low, high = mean_interval(_paired(new_report, old_report, order, name))
        metrics[name] = {"conclusion": None, "high": high, "low": low, "mean_diff": mean}
    new_gap = _number(_aggregate(new_report)["profile_gap"])
    old_gap = _number(_aggregate(old_report)["profile_gap"])
    metrics["profile_gap"] = {
        "conclusion": None,
        "diff": new_gap - old_gap,
        "new": new_gap,
        "old": old_gap,
    }
    return {"metrics": metrics, "n": len(order)}


def probe_block(
    rows: Sequence[Mapping[str, object]],
    inputs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """How close each output stays to its input, overall and by kind of input."""
    similarity: list[float] = []
    length_ratio: list[float] = []
    by_category: dict[str, list[float]] = defaultdict(list)
    identical = 0
    for row in rows:
        doc_id = str(row["id"])
        source = inputs.get(doc_id)
        if source is None:
            raise SystemExit(f"探针输出里有输入文件之外的 id: {doc_id}")
        text = str(source["vernacular"])
        output = str(row["output"])
        value = text_similarity(output, text)
        similarity.append(value)
        length_ratio.append(len(output) / len(text))
        by_category[str(source.get("category", "all"))].append(value)
        identical += int(output == text)
    if len(similarity) != len(inputs):
        raise SystemExit("探针输出条数与输入不一致")
    return {
        "by_category": {
            name: {"mean": sum(values) / len(values), "n": len(values)}
            for name, values in sorted(by_category.items())
        },
        "identical": identical,
        "length_ratio": summarize(length_ratio),
        "n": len(similarity),
        "near_copies": sum(1 for value in similarity if value >= NEAR_COPY),
        "similarity": summarize(similarity),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线评估换基座后的结果并写汇总（SPEC 4.10）")
    parser.add_argument("--tag", default="14b", help="run_model_size 用的规模标签")
    parser.add_argument("--generations-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument("--probes", type=Path, default=Path("eval/probes/modern_inputs.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("eval/configs/eval59.yaml"))
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("eval/.judge_cache"))
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None, *, timestamp: str | None = None) -> dict[str, object]:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    tag = str(args.tag)
    config = load_config(args.config)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    keys = {
        "base_new": f"base-{tag}-eval59",
        "lora_new": f"lora-{tag}-eval59",
        "base_old": "base-eval59",
        "lora_old": "lora-eval59",
    }
    reports: dict[str, dict[str, object]] = {}
    for role in ("base_new", "lora_new"):
        path = args.generations_dir / f"{keys[role]}.jsonl"
        if not path.is_file():
            raise SystemExit(f"生成文件不存在: {path}")
        body = evaluate(
            config,
            path,
            keys[role],
            skip_judge=True,
            report_path=args.reports_dir / f"{keys[role]}.json",
            cache_dir=args.cache_dir,
            completer=None,
        )
        reports[role] = body.model_dump()
    for role in ("base_old", "lora_old"):
        reports[role] = _existing_report(
            args.reports_dir / f"{keys[role]}.json",
            args.generations_dir / f"{keys[role]}.jsonl",
        )
    inputs = {str(row["id"]): row for row in _read_rows(args.probes)}
    probes = {
        name: probe_block(_read_rows(args.generations_dir / f"{name}.jsonl"), inputs)
        for name in ("base-3b-probe", "lora-3b-probe", f"base-{tag}-probe", f"lora-{tag}-probe")
    }
    summary = assemble_summary(
        tag=tag,
        keys=keys,
        reports=reports,
        probes=probes,
        timestamp=timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    stem = args.reports_dir / f"model-size-{tag}"
    stem.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(str(summary["changelog_markdown"]), encoding="utf-8")
    LOGGER.info("model_size_eval_written", summary=str(stem.with_suffix(".json")))
    return summary


def assemble_summary(
    *,
    tag: str,
    keys: Mapping[str, str],
    reports: Mapping[str, Mapping[str, object]],
    probes: Mapping[str, Mapping[str, object]],
    timestamp: str,
) -> dict[str, object]:
    order = _case_order(reports["lora_old"])
    for role, report in reports.items():
        if _case_order(report) != order:
            raise SystemExit(f"样本顺序不一致: {keys[role]}")
    summary: dict[str, object] = {
        "changelog_markdown": "",
        "groups": [
            {
                "aggregate": {
                    name: _aggregate(reports[role]).get(name) for name in AGGREGATE_NAMES
                },
                "name": keys[role],
                "sentence_length_delta": _sentence_delta(reports[role]),
            }
            for role in ("base_old", "lora_old", "base_new", "lora_new")
        ],
        "primary": {
            "new": keys["lora_new"],
            "old": keys["lora_old"],
            **compare(reports["lora_new"], reports["lora_old"], order),
        },
        "probe_note": PROBE_NOTE,
        "probes": dict(probes),
        "secondary": {
            "fine_tuning_on_new_base": {
                "new": keys["lora_new"],
                "old": keys["base_new"],
                **compare(reports["lora_new"], reports["base_new"], order),
            },
            "base_size": {
                "new": keys["base_new"],
                "old": keys["base_old"],
                **compare(reports["base_new"], reports["base_old"], order),
            },
        },
        "spec": "4.10",
        "tag": tag,
        "timestamp": timestamp,
    }
    summary["changelog_markdown"] = render_markdown(summary)
    return summary


def render_markdown(summary: Mapping[str, object]) -> str:
    primary = _obj(summary["primary"])
    lines = [
        f"主比较：{primary['new']} 对 {primary['old']}"
        f"（同样 {primary['n']} 条，逐条配对，新 − 旧）。"
        f"规则见 SPEC {summary['spec']}。只改了基座，其余设置相同。",
        "",
    ]
    lines.extend(_comparison_table(primary))
    for title, name in (
        ("次要：在新基座上微调是否仍有用", "fine_tuning_on_new_base"),
        ("次要：不微调时基座规模的差别", "base_size"),
    ):
        block = _obj(_obj(summary["secondary"])[name])
        lines.extend(["", f"**{title}（{block['new']} − {block['old']}）**", ""])
        lines.extend(_comparison_table(block))
    lines.extend(
        [
            "",
            "| 组合 | style_distance | profile_gap | profile_gap_per_case | entity_recall "
            "| numeral_recall | hallucination_rate | copy_ratio "
            "| sent_len_std 差 | sent_len_p90 差 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in _list(summary["groups"]):
        group = _obj(item)
        aggregate = _obj(group["aggregate"])
        delta = _obj(group["sentence_length_delta"])
        cells = [str(group["name"])]
        cells.extend(_num(aggregate[name]) for name in AGGREGATE_NAMES)
        cells.extend(_num(delta[name]) for name in SENTENCE_FEATURES)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "**探针集（探索性，不下结论）**",
            "",
            str(summary["probe_note"]),
            "",
            f"| 模型 | 条数 | 与输入相似度均值 | 中位数 | 相似度 ≥ {NEAR_COPY} 的条数 "
            "| 与输入完全相同 | 字数比均值 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    categories: list[str] = []
    for name, value in _obj(summary["probes"]).items():
        block = _obj(value)
        similarity = _obj(block["similarity"])
        lines.append(
            f"| {name} | {block['n']} | {_num(similarity['mean'])} | {_num(similarity['median'])} "
            f"| {block['near_copies']} | {block['identical']} "
            f"| {_num(_obj(block['length_ratio'])['mean'])} |"
        )
        categories = list(_obj(block["by_category"]))
    lines.extend(
        ["", "| 模型 | " + " | ".join(categories) + " |", "|---|" + "---|" * len(categories)]
    )
    for name, value in _obj(summary["probes"]).items():
        by_category = _obj(_obj(value)["by_category"])
        cells = [_num(_obj(by_category[category])["mean"]) for category in categories]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "上表各列是按输入类型分组的相似度均值。", ""])
    return "\n".join(lines)


def _comparison_table(block: Mapping[str, object]) -> list[str]:
    lines = ["| 指标 | 均值差 | 95% 区间 | 结论 |", "|---|---|---|---|"]
    for name, cell in _obj(block["metrics"]).items():
        item = _obj(cell)
        if "mean_diff" in item:
            conclusion = item["conclusion"] if item["conclusion"] is not None else "只报告"
            interval = f"[{_num(item['low'])}, {_num(item['high'])}]"
            lines.append(f"| {name} | {_num(item['mean_diff'])} | {interval} | {conclusion} |")
        else:
            lines.append(f"| {name} | {_num(item['diff'])} | — | 只报点估计 |")
    return lines


def _existing_report(report_path: Path, generations_path: Path) -> dict[str, object]:
    """A committed 3B report, checked against the generation file it claims to score."""
    if not report_path.is_file():
        raise SystemExit(f"既有报告不存在: {report_path}")
    report = _obj(json.loads(report_path.read_text(encoding="utf-8")))
    outputs = {str(row["id"]): str(row["output"]) for row in _read_rows(generations_path)}
    for item in _list(report["per_case"]):
        case = _obj(item)
        doc_id = str(case["id"])
        if doc_id not in outputs or output_hash(outputs[doc_id]) != case["output_hash"]:
            raise SystemExit(f"报告与生成文件对不上: {report_path.name} {doc_id}")
    return report


def _sentence_delta(report: Mapping[str, object]) -> dict[str, float]:
    delta = _obj(report.get("profile_mean_delta"))
    return {name: _number(delta[name]) for name in SENTENCE_FEATURES}


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path}")
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


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit("汇总结构损坏：不是数")
    return float(value)


if __name__ == "__main__":
    main()
