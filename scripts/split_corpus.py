"""Create and apply the immutable, work-stratified train/eval split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import structlog
from pydantic import BaseModel

from scripts.chunk_corpus import CorpusChunk

LOGGER = structlog.get_logger()
DEFAULT_SEED = 42
DEFAULT_EVAL_RATIO = 0.08


class CorpusSplit(BaseModel):
    """Stable split artifact containing IDs only."""

    seed: int
    eval_ratio: float
    train: list[str]
    eval: list[str]


def stratified_split(
    chunks: list[CorpusChunk], *, seed: int = DEFAULT_SEED, eval_ratio: float = DEFAULT_EVAL_RATIO
) -> CorpusSplit:
    """Split each work independently with a deterministic PRNG."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.work].append(chunk.id)

    eval_ids: set[str] = set()
    for work in sorted(grouped):
        ids = sorted(grouped[work])
        rng = random.Random(f"{seed}:{work}")
        rng.shuffle(ids)
        count = max(1, round(len(ids) * eval_ratio))
        eval_ids.update(ids[:count])

    all_ids = [chunk.id for chunk in chunks]
    return CorpusSplit(
        seed=seed,
        eval_ratio=eval_ratio,
        train=[chunk_id for chunk_id in all_ids if chunk_id not in eval_ids],
        eval=[chunk_id for chunk_id in all_ids if chunk_id in eval_ids],
    )


def load_chunks(path: Path) -> list[CorpusChunk]:
    with path.open(encoding="utf-8") as handle:
        return [CorpusChunk.model_validate_json(line) for line in handle if line.strip()]


def write_split(split: CorpusSplit, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(split.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按作品分层并固定 train/eval 划分")
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = stratified_split(load_chunks(args.chunks), seed=args.seed, eval_ratio=args.eval_ratio)
    write_split(split, args.output)
    LOGGER.info(
        "corpus_split_written",
        train=len(split.train),
        eval=len(split.eval),
        seed=split.seed,
        output=str(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
