"""Train one larger base and generate everything SPEC 4.10 compares, on a rented GPU.

Each step runs in its own process so the GPU is released between them, and a step
whose output already exists is skipped, so a dropped connection costs one step.
At the end the generations, the loss curve and a manifest are packed into one
archive to download. The adapter stays on the machine; it is large.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import structlog
from pydantic import BaseModel

from scripts.train import BASE_MODEL, EXPECTED_TRAINABLE_BY_MODEL

LOGGER = structlog.get_logger()
TAGS = {
    BASE_MODEL: "3b",
    "Qwen/Qwen2.5-7B-Instruct": "7b",
    "Qwen/Qwen2.5-14B-Instruct": "14b",
}
PACKAGES = ("torch", "transformers", "unsloth", "peft", "trl", "bitsandbytes")


class Step(BaseModel):
    """One subprocess and the file that proves it finished."""

    name: str
    command: list[str]
    output: Path


def plan_steps(
    *,
    base_model: str,
    pairs: Path,
    probes: Path,
    adapters_dir: Path,
    generations_dir: Path,
    python: str,
) -> list[Step]:
    """Train, then four generations: base and fine-tuned, on eval and on the probes."""
    tag = TAGS[base_model]
    run_id = f"sutong-v2-{tag}"
    adapter = adapters_dir / run_id / "adapter"

    def generate(name: str, source: Path, split: str, tuned: bool) -> Step:
        command = [python, "-m", "scripts.generate", "--pairs", str(source), "--split", split]
        command += ["--run-id", name, "--output", str(generations_dir / f"{name}.jsonl")]
        command += ["--adapter", str(adapter)] if tuned else ["--base-model", base_model]
        return Step(name=name, command=command, output=generations_dir / f"{name}.jsonl")

    return [
        Step(
            name="train",
            command=[
                python,
                "-m",
                "scripts.train",
                "--pairs",
                str(pairs),
                "--base-model",
                base_model,
                "--run-id",
                run_id,
                "--output-dir",
                str(adapters_dir),
            ],
            output=adapters_dir / run_id / "loss_curve.json",
        ),
        generate(f"base-{tag}-eval59", pairs, "eval", tuned=False),
        generate(f"lora-{tag}-eval59", pairs, "eval", tuned=True),
        generate(f"base-{tag}-probe", probes, "probe", tuned=False),
        generate(f"lora-{tag}-probe", probes, "probe", tuned=True),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="换基座训练并生成 SPEC 4.10 需要的全部输出")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-14B-Instruct",
        choices=sorted(EXPECTED_TRAINABLE_BY_MODEL),
    )
    parser.add_argument("--pairs", type=Path, default=Path("corpus/pairs.jsonl"))
    parser.add_argument("--probes", type=Path, default=Path("eval/probes/modern_inputs.jsonl"))
    parser.add_argument("--adapters-dir", type=Path, default=Path("adapters"))
    parser.add_argument("--generations-dir", type=Path, default=Path("corpus/generations"))
    parser.add_argument("--reports-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--dry-run", action="store_true", help="只打印要执行的步骤，不运行")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    tag = TAGS[args.base_model]
    steps = plan_steps(
        base_model=args.base_model,
        pairs=args.pairs,
        probes=args.probes,
        adapters_dir=args.adapters_dir,
        generations_dir=args.generations_dir,
        python=sys.executable,
    )
    if args.dry_run:
        for step in steps:
            LOGGER.info("planned_step", name=step.name, command=" ".join(step.command))
        return 0
    for path in (args.pairs, args.probes):
        if not path.is_file():
            raise SystemExit(f"输入文件不存在: {path}")
    seconds: dict[str, float | None] = {}
    for step in steps:
        if step.output.is_file():
            LOGGER.info("step_skipped", name=step.name, output=str(step.output))
            seconds[step.name] = None
            continue
        LOGGER.info("step_start", name=step.name)
        started = time.perf_counter()
        # A failed step stops the run: later steps would read a missing adapter.
        subprocess.run(step.command, check=True)
        seconds[step.name] = round(time.perf_counter() - started, 1)
        if not step.output.is_file():
            raise SystemExit(f"步骤结束但没有产出: {step.name} {step.output}")
        LOGGER.info("step_done", name=step.name, seconds=seconds[step.name])
    curve = steps[0].output
    manifest = {
        "base_model": args.base_model,
        "commit": _git("rev-parse", "HEAD"),
        "files": {
            step.name: {
                "path": step.output.as_posix(),
                "sha256": hashlib.sha256(step.output.read_bytes()).hexdigest(),
            }
            for step in steps
        },
        "gpu": _gpu(),
        "loss_curve": json.loads(curve.read_text(encoding="utf-8")),
        "packages": {name: _version(name) for name in PACKAGES},
        # None means the step's output was already there and it did not run again.
        "seconds": seconds,
        "tag": tag,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reports_dir / f"model-size-{tag}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = Path(f"model-size-{tag}-results.tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(manifest_path, arcname=manifest_path.as_posix())
        for step in steps[1:]:
            handle.add(step.output, arcname=step.output.as_posix())
    LOGGER.info("model_size_complete", archive=str(archive), manifest=str(manifest_path))
    return 0


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _gpu() -> str | None:
    try:
        query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        return subprocess.check_output(query, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
