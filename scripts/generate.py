"""Generate stylized text for a split, with or without a LoRA adapter.

The prompt shape is imported from scripts.train, never re-typed here: if training
and inference disagree by one token the adapter is worthless. Decoding is greedy
so a run is reproducible from (adapter, seed, prompt) alone.

Heavy imports live inside functions so --help works without a GPU stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import BASE_MODEL, Pair, build_messages, load_pairs, select  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

LOGGER = structlog.get_logger()
MAX_NEW_TOKENS = 768


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用基座或 LoRA adapter 批量生成")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--split", default="eval")
    parser.add_argument("--adapter", type=Path, default=None, help="不给就是无微调基座")
    parser.add_argument("--pipeline", default="baseline")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，用于冒烟")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def model_name(adapter: Path | None) -> str:
    """What goes into the generation row's model field."""
    return adapter.parent.name if adapter is not None else "base"


def greedy_decode(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Greedy decode of one chat. sweep_topk.py must call this, not a copy."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def generate_one(
    model: Any,
    tokenizer: Any,
    vernacular: str,
    max_new_tokens: int,
) -> str:
    """Greedy decode. The prompt comes from scripts.train.build_messages."""
    return greedy_decode(model, tokenizer, build_messages(vernacular, None), max_new_tokens)


def load_inference_model(adapter: Path | None, max_seq_length: int, seed: int) -> tuple[Any, Any]:
    """Load the base or a LoRA adapter once. Imports stay inside the call."""
    import torch
    from unsloth import FastLanguageModel

    torch.manual_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter) if adapter is not None else BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])

    cases: list[Pair] = select(load_pairs(args.pairs), args.split)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError(f"{args.pairs} 里没有 split == {args.split!r} 的样本")

    output = args.output or Path("corpus/generations") / f"{args.run_id}.jsonl"
    name = model_name(args.adapter)
    LOGGER.info(
        "generate_start",
        run_id=args.run_id,
        model=name,
        adapter=str(args.adapter) if args.adapter else None,
        pipeline=args.pipeline,
        cases=len(cases),
        seed=args.seed,
        decoding="greedy",
    )

    model, tokenizer = load_inference_model(args.adapter, args.max_seq_length, args.seed)

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        rows.append(
            {
                "id": case.id,
                "model": name,
                "pipeline": args.pipeline,
                "output": generate_one(model, tokenizer, case.vernacular, args.max_new_tokens),
                "trace": None,
                "seed": args.seed,
            }
        )
        if index % 10 == 0 or index == len(cases):
            LOGGER.info("generate_progress", done=index, total=len(cases))

    write_rows(output, rows)
    LOGGER.info("generate_complete", output=str(output), rows=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
