"""Learn the literary lexicon from train pairs. Words and scores only."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import jieba
import structlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stylometry.lexicon import build_lexicon, lexicon_payload  # noqa: E402

LOGGER = structlog.get_logger()
CJK = re.compile(r"[\u3400-\u9fff]")


def countable(text: str) -> list[str]:
    """Content tokens. Punctuation and non-Chinese fragments stay out of the lexicon."""
    return [token for token in jieba.lcut(text) if CJK.search(token)]


def load_train_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload["train"])


def main() -> None:
    parser = argparse.ArgumentParser(description="从 train pair 统计书面语和口语词表")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("corpus/split.json"))
    parser.add_argument("--output", type=Path, default=Path("stylometry/data/lexicon.json"))
    args = parser.parse_args()
    train_ids = load_train_ids(args.split)
    original_counts: Counter[str] = Counter()
    vernacular_counts: Counter[str] = Counter()
    used = 0
    skipped = 0
    with args.pairs.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in train_ids:
                skipped += 1
                continue
            original_counts.update(countable(row["original"]))
            vernacular_counts.update(countable(row["vernacular"]))
            used += 1
    lexicon = build_lexicon(original_counts, vernacular_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lexicon_payload(lexicon), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "lexicon_built",
        train_pairs=used,
        held_out_pairs=skipped,
        literary=len(lexicon.literary),
        colloquial=len(lexicon.colloquial),
        output=str(args.output),
    )


if __name__ == "__main__":
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    main()
