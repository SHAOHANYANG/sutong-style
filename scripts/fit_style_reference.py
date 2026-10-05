"""Fit the Su Tong style reference on every original chunk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stylometry.distance import StyleReference  # noqa: E402
from stylometry.features import FEATURE_NAMES  # noqa: E402
from stylometry.lexicon import LiteraryLexicon  # noqa: E402

LOGGER = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="用全部原文拟合风格参照的均值和标准差")
    parser.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
    parser.add_argument("--lexicon", type=Path, default=Path("stylometry/data/lexicon.json"))
    parser.add_argument("--output", type=Path, default=Path("stylometry/data/style_reference.json"))
    args = parser.parse_args()
    lexicon = LiteraryLexicon.model_validate_json(args.lexicon.read_text(encoding="utf-8"))
    texts = [
        json.loads(line)["original"]
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference = StyleReference.fit(texts, lexicon)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "n_texts": len(texts),
        "mean": reference.mean.tolist(),
        "std": reference.std.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("style_reference_fitted", n_texts=len(texts), output=str(args.output))


if __name__ == "__main__":
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    main()
