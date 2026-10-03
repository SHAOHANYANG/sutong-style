"""Turn original chunks into conservative plan-A vernacular training inputs."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

import jieba.posseg as pseg
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel

from scripts.chunk_corpus import CorpusChunk, cjk_length
from scripts.split_corpus import CorpusSplit

LOGGER = structlog.get_logger()
OUTPUT_LOCK = threading.Lock()
MIN_RATIO = 0.9
MAX_RATIO = 1.3
STRICT_COPY_SIMILARITY = 0.65
MIXED_COPY_SIMILARITY = 0.68
MAX_COPY_SIMILARITY = 0.70
HIGH_DIALOGUE_DENSITY = 0.50
MIXED_DIALOGUE_DENSITY = 0.20
HIGH_PROPER_NOUN_DENSITY = 0.10
MIXED_PROPER_NOUN_DENSITY = 0.07
PROPER_NOUN_FLAGS = {"nr", "nrfg", "nrt", "ns", "nt", "nz"}
DIALOGUE_UNIT = re.compile(r"[“”「」『』]|(?:说|问|答|喊|嚷|道)[，。！？：]")
NUMERAL_PHRASE = re.compile(
    r"(?:差不多|将近|第|初|近|约|上)?"
    r"(?:[0-9０-９]+|[零〇一二两三四五六七八九十百千万亿]+)"
    r"(?:公斤|公里|大洋|年|月|日|号|岁|个|位|名|人|口|只|条|件|家|间|所|匹|头|斤|两|克|"
    r"里|亩|州|县|国|军|代|届|次|回|遍|天|夜|块|元|角|分|束|朵|颗|粒|枚|辆|艘|顶|把|柄|"
    r"张|页|本|封|桌|场|杯|碗|盘)"
)
WIDTH_TRANSLATION = str.maketrans("０１２３４５６７８９", "0123456789")
NUMERAL_VALUE = re.compile(
    r"^(?P<modifier>差不多|将近|第|初|近|约|上)?"
    r"(?P<number>[0-9]+|[零〇一二两三四五六七八九十百千万亿]+)"
)
GRAMMATICAL_ONE = re.compile(r"^一(?:个|只|口|把|件|家|间|条|位|张|片)$")


class Generator(Protocol):
    """Injected text generation boundary."""

    def generate(self, prompt: str, **kwargs: object) -> str: ...


class Pair(BaseModel):
    """One reconstructed parallel pair."""

    id: str
    work: str
    idx: int
    vernacular: str
    original: str
    split: str


class VernacularExample(BaseModel):
    """A local-only plan-A style demonstration loaded from salvaged data."""

    id: str
    original: str
    vernacular: str


class AttemptIssue(BaseModel):
    id: str
    attempt: int
    reason: str
    length_ratio: float | None = None
    similarity: float | None = None
    similarity_limit: float | None = None
    dialogue_density: float | None = None
    proper_noun_density: float | None = None
    expected_numerals: list[str] | None = None
    actual_numerals: list[str] | None = None


class RebuildReport(BaseModel):
    total_requested: int
    already_done: int
    succeeded: int
    completed_after_run: int
    failed: list[str]
    out_of_range_attempts: list[AttemptIssue]
    copy_similarity_exceptions: list[AttemptIssue]
    attempt_issues: list[AttemptIssue]


PROMPT_TEMPLATE = """你在重建一份“现代白话 → 文学原文”的平行语料。
请把下面的文学原文彻底改写成现代、自然、易懂的口头白话。
采用方案 A：参照完整旧样本的改写强度，保留每段的全部信息和情节顺序。
先理解这一段发生了什么，再用自己的日常说法重新讲出来。
段落结构保持一致，但段内可以拆句、合句、调整主谓宾和定语位置。

硬性要求：
1. 人名、地名、职官称谓、数字、日期、数量必须逐一保留。
2. 情节、动作、因果、对话内容不得删减、概括或增补。
3. 原文几段，白话也必须几段；段落顺序不变。
4. 改写句子骨架和叙述方式，而不只是逐词换同义词。长句拆成短句，
   定语改成独立说明，书面语、成语、文言残留换成日常口语。
5. 原文中没有引号的对话，白话中改用现代常规引号。
6. 白话中文字数必须为原文的 0.9–1.3 倍。
7. 只输出白话正文，不要解释，不要标题，不要 Markdown 围栏。

改写强度示例（例句是虚构的）：
- “他未曾料到客人会来”改成“他没想到客人会来”
- “她不无惆怅地凝望庭院”改成“她看着院子，心里挺难受”
- “他径直前往，沉默良久”改成“他直接走过去，好一会儿没说话”

只添加引号、标点或替换少量词语不算白话改写，会被判定失败并要求重做。
不要概括，不要加原文没有的闲聊；改变的是表达方式，不是事实和内容。
人名、地名、称谓、数字按原样保留；它们周围的描述和对白表达仍应重新组织。

原文：
{original}
"""


def numeric_phrases(text: str) -> Counter[str]:
    """Extract quantity phrases while normalizing full-width Arabic digits."""
    return Counter(
        match.group(0).translate(WIDTH_TRANSLATION) for match in NUMERAL_PHRASE.finditer(text)
    )


def numeric_facts(text: str) -> set[str]:
    """Normalize numeric values while ignoring interchangeable classifiers."""
    facts: set[str] = set()
    for phrase in numeric_phrases(text):
        if GRAMMATICAL_ONE.fullmatch(phrase):
            continue
        match = NUMERAL_VALUE.match(phrase)
        if not match:
            continue
        modifier = match.group("modifier") or ""
        modifier = {"将近": "近", "差不多": "约"}.get(modifier, modifier)
        number = match.group("number")
        if len(number) > 1 and number[0] == "一" and number[1] in "十百千万":
            number = number[1:]
        facts.add(modifier + number)
    return facts


def numerals_preserved(original: str, vernacular: str) -> bool:
    """Require every input quantity phrase without rejecting harmless new phrases."""
    return numeric_facts(original).issubset(numeric_facts(vernacular))


def build_prompt(
    original: str,
    examples: list[VernacularExample] | None = None,
    feedback: str = "",
    previous_output: str = "",
) -> str:
    demonstrations = ""
    if examples:
        rendered = [
            f"范例 {index} 原文：\n{example.original}\n范例 {index} 白话：\n{example.vernacular}"
            for index, example in enumerate(examples, start=1)
        ]
        demonstrations = (
            "下面是旧流程留下的方案 A 范例。模仿它的白话强度，"
            "但不要借用范例中的事实：\n\n" + "\n\n".join(rendered) + "\n\n"
        )
    retry_instruction = ""
    if feedback:
        retry_instruction = (
            f"上一次结果未通过校验：{feedback}\n"
            "下面的上一版仅用于定位问题。重新从原文组织表达，保留事实，"
            f"不要沿用上一版的句子骨架：\n{previous_output}\n\n"
        )
    return demonstrations + retry_instruction + PROMPT_TEMPLATE.format(original=original)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped


def paragraph_count(text: str) -> int:
    """Count non-empty paragraphs represented by separate lines."""
    return sum(bool(line.strip()) for line in text.splitlines())


def text_similarity(original: str, vernacular: str) -> float:
    """Measure direct copy similarity using the salvaged-reference metric."""
    return difflib.SequenceMatcher(None, vernacular, original).ratio()


def select_style_examples(
    examples: list[VernacularExample], *, chunk_id: str, attempt: int, seed: int
) -> list[VernacularExample]:
    """Rotate 2-3 complete demonstrations deterministically, excluding the target."""
    eligible = [example for example in examples if example.id != chunk_id]
    random.Random(f"{seed}:{chunk_id}").shuffle(eligible)
    if not eligible:
        return []
    count = min(2 + attempt % 2, len(eligible))
    start = 3 * (attempt - 1)
    return [eligible[(start + offset) % len(eligible)] for offset in range(count)]


def dialogue_density(text: str) -> float:
    """Estimate the CJK-character share occupied by dialogue-bearing units."""
    total = cjk_length(text)
    if not total:
        return 0.0
    units = re.split(r"(?<=[。！？；])|\n", text)
    dialogue_chars = sum(cjk_length(unit) for unit in units if DIALOGUE_UNIT.search(unit))
    return dialogue_chars / total


def proper_noun_density(text: str) -> float:
    """Estimate the CJK-character share tagged as names, places, or organizations."""
    total = cjk_length(text)
    if not total:
        return 0.0
    proper_chars = sum(
        cjk_length(word) for word, flag in pseg.cut(text) if flag in PROPER_NOUN_FLAGS
    )
    return proper_chars / total


def copy_similarity_limit(text: str) -> float:
    """Choose a reference-calibrated threshold from original-text densities."""
    dialogue = dialogue_density(text)
    proper_nouns = proper_noun_density(text)
    if dialogue >= HIGH_DIALOGUE_DENSITY or proper_nouns >= HIGH_PROPER_NOUN_DENSITY:
        return MAX_COPY_SIMILARITY
    if dialogue >= MIXED_DIALOGUE_DENSITY or proper_nouns >= MIXED_PROPER_NOUN_DENSITY:
        return MIXED_COPY_SIMILARITY
    return STRICT_COPY_SIMILARITY


def generate_pair(
    chunk: CorpusChunk,
    split_by_id: dict[str, str],
    generator: Generator,
    *,
    retries: int,
    seed: int,
    examples: list[VernacularExample] | None = None,
) -> tuple[Pair | None, list[AttemptIssue]]:
    """Generate one valid-length pair, retrying failures without aborting the batch."""
    issues: list[AttemptIssue] = []
    original_length = cjk_length(chunk.original)
    feedback = ""
    previous_output = ""
    chunk_dialogue_density = dialogue_density(chunk.original)
    chunk_proper_noun_density = proper_noun_density(chunk.original)
    similarity_limit = copy_similarity_limit(chunk.original)
    lowest_over_limit: float | None = None
    for attempt in range(1, retries + 2):
        try:
            vernacular = _strip_fences(
                generator.generate(
                    build_prompt(
                        chunk.original,
                        select_style_examples(
                            examples or [], chunk_id=chunk.id, attempt=attempt, seed=seed
                        ),
                        feedback,
                        previous_output,
                    ),
                    seed=seed + chunk.idx + attempt,
                )
            )
            ratio = cjk_length(vernacular) / original_length if original_length else 0.0
            previous_output = vernacular
            if not MIN_RATIO <= ratio <= MAX_RATIO:
                issues.append(
                    AttemptIssue(
                        id=chunk.id,
                        attempt=attempt,
                        reason="length_out_of_range",
                        length_ratio=round(ratio, 4),
                    )
                )
                feedback = "字数不在原文的 0.9–1.3 倍内，请在完整保留信息的前提下调整长度。"
                continue
            if paragraph_count(vernacular) != paragraph_count(chunk.original):
                issues.append(
                    AttemptIssue(
                        id=chunk.id,
                        attempt=attempt,
                        reason="paragraph_count_mismatch",
                        length_ratio=round(ratio, 4),
                    )
                )
                feedback = (
                    f"段落数必须与原文同为 {paragraph_count(chunk.original)} 段，不要拆段或合段。"
                )
                continue
            if not numerals_preserved(chunk.original, vernacular):
                issues.append(
                    AttemptIssue(
                        id=chunk.id,
                        attempt=attempt,
                        reason="numeral_mismatch",
                        length_ratio=round(ratio, 4),
                        expected_numerals=sorted(numeric_facts(chunk.original)),
                        actual_numerals=sorted(numeric_facts(vernacular)),
                    )
                )
                protected = [match.group(0) for match in NUMERAL_PHRASE.finditer(chunk.original)]
                feedback = (
                    "下列数字短语被遗漏或改动，必须原样保留："
                    + "、".join(protected)
                    + "。例如“万人”不能改成“上万人”。"
                )
                continue
            similarity = text_similarity(chunk.original, vernacular)
            if similarity > similarity_limit:
                if lowest_over_limit is None or similarity < lowest_over_limit:
                    lowest_over_limit = similarity
                issues.append(
                    AttemptIssue(
                        id=chunk.id,
                        attempt=attempt,
                        reason="insufficient_rewrite",
                        length_ratio=round(ratio, 4),
                        similarity=round(similarity, 4),
                        similarity_limit=similarity_limit,
                        dialogue_density=round(chunk_dialogue_density, 4),
                        proper_noun_density=round(chunk_proper_noun_density, 4),
                    )
                )
                feedback = (
                    f"序列相似度 {similarity:.3f} 超过本段上限 {similarity_limit:.2f}。"
                    "不能只加标点或替换少量词；请改写句子骨架，把书面措辞换成口头白话。"
                )
                continue
            pair = Pair(
                id=chunk.id,
                work=chunk.work,
                idx=chunk.idx,
                vernacular=vernacular,
                original=chunk.original,
                split=split_by_id.get(chunk.id, "train"),
            )
            return pair, issues
        except Exception as exc:
            issues.append(AttemptIssue(id=chunk.id, attempt=attempt, reason=type(exc).__name__))
    if lowest_over_limit is not None:
        issues.append(
            AttemptIssue(
                id=chunk.id,
                attempt=retries + 1,
                reason="copy_similarity_exception",
                similarity=round(lowest_over_limit, 4),
                similarity_limit=similarity_limit,
                dialogue_density=round(chunk_dialogue_density, 4),
                proper_noun_density=round(chunk_proper_noun_density, 4),
            )
        )
    return None, issues


def load_chunks(path: Path) -> list[CorpusChunk]:
    with path.open(encoding="utf-8") as handle:
        return [CorpusChunk.model_validate_json(line) for line in handle if line.strip()]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {Pair.model_validate_json(line).id for line in handle if line.strip()}


def load_split(path: Path) -> dict[str, str]:
    split = CorpusSplit.model_validate_json(path.read_text(encoding="utf-8"))
    return {**dict.fromkeys(split.train, "train"), **dict.fromkeys(split.eval, "eval")}


def load_preview_reference(path: Path) -> list[CorpusChunk]:
    """Load only the 14 complete salvaged originals for human A/B comparison."""
    chunks: list[CorpusChunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("maybe_truncated", True):
                chunks.append(
                    CorpusChunk(
                        id=row["id"], work=row["work"], idx=row["idx"], original=row["original"]
                    )
                )
    return chunks


def load_style_examples(path: Path) -> list[VernacularExample]:
    """Load short complete examples without embedding copyrighted text in source code."""
    examples: list[VernacularExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("maybe_truncated", True):
                examples.append(
                    VernacularExample(
                        id=row["id"], original=row["original"], vernacular=row["vernacular"]
                    )
                )
    return sorted(examples, key=lambda example: cjk_length(example.original))


def select_work_balanced_fill(
    chunks: list[CorpusChunk], *, count: int, excluded_ids: set[str]
) -> list[CorpusChunk]:
    selected: list[CorpusChunk] = []
    by_work: dict[str, list[CorpusChunk]] = {}
    for chunk in chunks:
        if chunk.id not in excluded_ids:
            by_work.setdefault(chunk.work, []).append(chunk)
    while len(selected) < count and any(by_work.values()):
        for work in sorted(by_work):
            if by_work[work] and len(selected) < count:
                candidates = by_work[work]
                selected.append(candidates.pop(len(candidates) // 2))
    return selected


def run_batch(
    chunks: list[CorpusChunk],
    output: Path,
    report_path: Path,
    split_by_id: dict[str, str],
    generator: Generator,
    *,
    concurrency: int = 4,
    retries: int = 2,
    seed: int = 42,
    examples: list[VernacularExample] | None = None,
) -> RebuildReport:
    """Run a resumable concurrent batch and persist every completed pair immediately."""
    completed = load_completed(output)
    remaining = [chunk for chunk in chunks if chunk.id not in completed]
    LOGGER.info(
        "vernacularize_start",
        total=len(chunks),
        already_done=len(completed),
        remaining=len(remaining),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed: list[str] = []
    issues: list[AttemptIssue] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[Future[tuple[Pair | None, list[AttemptIssue]]], CorpusChunk] = {
            executor.submit(
                generate_pair,
                chunk,
                split_by_id,
                generator,
                retries=retries,
                seed=seed,
                examples=examples,
            ): chunk
            for chunk in remaining
        }
        for processed, future in enumerate(as_completed(futures), start=1):
            chunk = futures[future]
            pair, pair_issues = future.result()
            issues.extend(pair_issues)
            if pair is None:
                failed.append(chunk.id)
            else:
                with OUTPUT_LOCK, output.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(pair.model_dump_json() + "\n")
                succeeded += 1
            if processed % 20 == 0 or processed == len(remaining):
                LOGGER.info(
                    "vernacularize_progress",
                    progress=f"{processed}/{len(remaining)}",
                    ok=succeeded,
                    failed=len(failed),
                )

    report = RebuildReport(
        total_requested=len(chunks),
        already_done=len(completed),
        succeeded=succeeded,
        completed_after_run=len(completed) + succeeded,
        failed=failed,
        out_of_range_attempts=[issue for issue in issues if issue.reason == "length_out_of_range"],
        copy_similarity_exceptions=[
            issue for issue in issues if issue.reason == "copy_similarity_exception"
        ],
        attempt_issues=issues,
    )
    report_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 OpenAI 兼容接口把原文降级改写成现代白话")
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--report", type=Path, default=Path("corpus/rebuild_report.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-reference", type=Path)
    parser.add_argument("--style-reference", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        LOGGER.error("missing_environment", variable="LLM_API_KEY")
        return 1

    from infra.openai_generator import OpenAICompatibleGenerator

    chunks = load_chunks(args.chunks)
    if args.preview_reference:
        preview = load_preview_reference(args.preview_reference)
        fill_count = max(0, (args.limit or len(preview)) - len(preview))
        preview.extend(
            select_work_balanced_fill(
                chunks, count=fill_count, excluded_ids={chunk.id for chunk in preview}
            )
        )
        chunks = preview
    elif args.limit is not None:
        chunks = chunks[: args.limit]

    generator = OpenAICompatibleGenerator(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("VERNACULARIZE_MODEL", "deepseek-chat"),
    )
    report = run_batch(
        chunks,
        args.output,
        args.report,
        load_split(args.split),
        generator,
        concurrency=args.concurrency,
        retries=args.retries,
        seed=args.seed,
        examples=load_style_examples(args.style_reference) if args.style_reference else None,
    )
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
