"""Build a committable name list from train originals. Words only, no sentences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.fidelity import build_entity_gazetteer  # noqa: E402

LOGGER = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="从 train 原文统计专名表")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--output", type=Path, default=Path("corpus/gazetteer.json"))
    args = parser.parse_args()
    train_ids = set(json.loads(args.split.read_text(encoding="utf-8"))["train"])
    texts: list[str] = []
    held_out = 0
    with args.pairs.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in train_ids:
                held_out += 1
                continue
            texts.append(row["original"])
    words = sorted(build_entity_gazetteer(texts))
    args.output.write_text(
        json.dumps({"words": words}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "gazetteer_built",
        train_texts=len(texts),
        held_out_pairs=held_out,
        words=len(words),
        output=str(args.output),
    )


if __name__ == "__main__":
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    main()
