"""Config and reference loaders. Must not import eval.judge: the agent run needs them on WSL2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel

from stylometry.distance import StyleReference
from stylometry.lexicon import LiteraryLexicon


class EvalConfig(BaseModel):
    """Committed baseline settings. Paths are relative to the working directory."""

    pipeline: str
    model: str
    seed: int
    cases: str
    style_reference: str
    lexicon: str
    gazetteer: str
    judge_model: str
    judge_prompt: str
    opponent: Literal["vernacular"]
    split: str | None = None
    """Keep only cases whose split matches. None evaluates the whole file."""


def load_config(path: Path) -> EvalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是映射：{path}")
    return EvalConfig.model_validate(payload)


def load_style_reference(config: EvalConfig) -> StyleReference:
    lexicon = LiteraryLexicon.model_validate_json(Path(config.lexicon).read_text(encoding="utf-8"))
    payload = json.loads(Path(config.style_reference).read_text(encoding="utf-8"))
    return StyleReference(
        mean=np.array(payload["mean"], dtype=np.float64),
        std=np.array(payload["std"], dtype=np.float64),
        lexicon=lexicon,
    )


def load_gazetteer(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload["words"]
    if not isinstance(words, list):
        raise ValueError("gazetteer words 不是列表")
    return {word for word in words if isinstance(word, str)}
