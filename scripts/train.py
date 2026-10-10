"""Fine-tune Qwen2.5-3B-Instruct into the Su Tong style with LoRA.

Hyper-parameters follow SPEC 1.2 with two deliberate corrections:

* ``r=16``, not the ``r=32`` written in the prose. The logged trainable-parameter
  count 29,933,568 is exactly r=16 over the seven target modules on this
  architecture; r=32 would be 59,867,136. The machine-printed count wins.
* ``epochs=2``, not 3. v1's eval loss bottomed at epoch 1.9 (2.056) and rose to
  2.089 by epoch 3. The third epoch was overfitting.

Epoch count is fixed before training starts, so the eval loss recorded here is
observational, not a model-selection signal.

Heavy imports live inside functions: ``--help`` and ``--dry-run`` must work on a
machine with no GPU stack installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

LOGGER = structlog.get_logger()

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_TRAINABLE = 29_933_568
# r = 16 on the seven modules below. Per layer that is
# 16 * (2 * 2h + 2 * (h + kv) + 3 * (h + m)), with kv = key/value heads * head size.
# A count that does not match means the LoRA config or the base differs from the plan.
EXPECTED_TRAINABLE_BY_MODEL = {
    BASE_MODEL: EXPECTED_TRAINABLE,
    "Qwen/Qwen2.5-7B-Instruct": 40_370_176,
    "Qwen/Qwen2.5-14B-Instruct": 68_812_800,
}
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
SYSTEM_PROMPT = "把下面的白话改写成苏童的文风。保留全部人名、地名、数字和事件，不要增删情节。"

V1_EVAL_LOSS = "2.169 → 2.056 (epoch 1.9 触底) → 2.089"


class TrainConfig(BaseModel):
    """Every number that defines the run. Printed verbatim by --dry-run."""

    base_model: str = BASE_MODEL
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = TARGET_MODULES
    epochs: int = 2
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 1536
    load_in_4bit: bool = True
    seed: int = 42
    expected_trainable: int = EXPECTED_TRAINABLE


class Pair(BaseModel):
    """One row of corpus/pairs.jsonl."""

    id: str
    work: str
    idx: int
    vernacular: str
    original: str
    split: str


def load_pairs(path: Path) -> list[Pair]:
    """Read the pair file. split.json is never consulted or rewritten here."""
    rows: list[Pair] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(Pair.model_validate_json(line))
        except ValueError as exc:  # pragma: no cover - malformed corpus
            raise ValueError(f"{path} 第 {line_number} 行格式错误: {exc}") from exc
    return rows


def select(rows: Sequence[Pair], split: str) -> list[Pair]:
    return [row for row in rows if row.split == split]


def build_messages(vernacular: str, original: str | None) -> list[dict[str, str]]:
    """The one true prompt shape.

    generate.py MUST call this with original=None and reuse the result, otherwise
    training and inference disagree and the adapter is worthless.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": vernacular},
    ]
    if original is not None:
        messages.append({"role": "assistant", "content": original})
    return messages


def render(tokenizer: Any, rows: Sequence[Pair]) -> list[str]:
    """Apply the chat template to every pair."""
    return [
        tokenizer.apply_chat_template(
            build_messages(row.vernacular, row.original),
            tokenize=False,
        )
        for row in rows
    ]


def assert_no_truncation(tokenizer: Any, texts: Sequence[str], limit: int) -> int:
    """A truncated target silently deletes supervision. Fail loudly instead."""
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    longest = max(lengths)
    if longest > limit:
        over = sum(1 for length in lengths if length > limit)
        raise ValueError(
            f"max_seq_length={limit} 会截断 {over} 条样本（最长 {longest} token）。"
            "调大 --max-seq-length，不要让目标被截断。"
        )
    return longest


def assert_trainable(model: Any, expected: int) -> int:
    """Pin the LoRA configuration to v1's logged parameter count."""
    actual = int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))
    if actual != expected:
        raise ValueError(
            f"可训练参数 {actual:,} 与预期 {expected:,} 不符。"
            "r、target_modules 或基座与 SPEC 1.2 不一致，先查清楚再训。"
        )
    return actual


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 微调 Qwen2.5 到苏童文风")
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--run-id", default="sutong-v2")
    parser.add_argument(
        "--base-model",
        default=None,
        choices=sorted(EXPECTED_TRAINABLE_BY_MODEL),
        help="默认 3B。换基座时其余超参不变（SPEC 4.10）",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("adapters"))
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认 2，一般不要动")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析配置和数据，不加载权重、不碰 GPU",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.max_seq_length is not None:
        overrides["max_seq_length"] = args.max_seq_length
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.base_model is not None:
        overrides["base_model"] = args.base_model
        overrides["expected_trainable"] = EXPECTED_TRAINABLE_BY_MODEL[args.base_model]
    return TrainConfig(**overrides)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolve_config(args)
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])

    rows = load_pairs(args.pairs)
    train_rows = select(rows, "train")
    eval_rows = select(rows, "eval")
    if not train_rows:
        raise ValueError(f"{args.pairs} 里没有 split == 'train' 的样本")

    LOGGER.info(
        "train_config",
        run_id=args.run_id,
        train=len(train_rows),
        eval=len(eval_rows),
        v1_eval_loss=V1_EVAL_LOSS,
        **config.model_dump(),
    )

    if args.dry_run:
        LOGGER.info("dry_run_complete", message="未加载权重，未使用 GPU")
        return 0

    # unsloth must be imported before trl/transformers/peft or its patches miss.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: skip
    from unsloth.chat_templates import train_on_responses_only  # isort: skip

    from datasets import Dataset  # isort: skip
    from trl import SFTConfig, SFTTrainer  # isort: skip

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.target_modules),
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    trainable = assert_trainable(model, config.expected_trainable)

    train_texts = render(tokenizer, train_rows)
    eval_texts = render(tokenizer, eval_rows)
    longest = assert_no_truncation(tokenizer, train_texts + eval_texts, config.max_seq_length)
    LOGGER.info("data_ready", trainable=trainable, longest_tokens=longest)

    train_set = Dataset.from_dict({"text": train_texts})
    eval_set = Dataset.from_dict({"text": eval_texts}) if eval_texts else None

    output_dir = args.output_dir / args.run_id
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_batch_size,
            per_device_eval_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler,
            warmup_ratio=config.warmup_ratio,
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="epoch" if eval_set is not None else "no",
            save_strategy="epoch",
            seed=config.seed,
            output_dir=str(output_dir),
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    history = [
        {key: value for key, value in entry.items() if key in {"epoch", "loss", "eval_loss"}}
        for entry in trainer.state.log_history
    ]
    curve_path = output_dir / "loss_curve.json"
    curve_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "config": config.model_dump(),
                "trainable": trainable,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "v1_eval_loss": V1_EVAL_LOSS,
                "history": [entry for entry in history if entry],
                "metrics": stats.metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "train_complete",
        adapter=str(adapter_dir),
        curve=str(curve_path),
        runtime_seconds=round(stats.metrics.get("train_runtime", 0.0), 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
