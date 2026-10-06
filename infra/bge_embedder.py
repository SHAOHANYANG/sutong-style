"""bge-m3 dense vectors via transformers. No FlagEmbedding, no sparse, no colbert."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 8


class BgeEmbedder:
    """CLS pooling of the last hidden state, then L2 normalization.

    Chunks are 200-400 characters, so 512 tokens covers them on an 8GB card.
    The index normalizes again; this normalization is part of the bge-m3 dense definition.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length 必须 >= 2")
        if batch_size < 1:
            raise ValueError("batch_size 必须是正整数")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncated = 0
        if device is None:
            device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype)
        self._model.to(device)
        self._model.eval()
        commit = getattr(self._model.config, "_commit_hash", None)
        if not isinstance(commit, str) or not commit:
            raise RuntimeError("模型没有返回 commit hash，拒绝继续")
        self.revision = commit
        hidden = getattr(self._model.config, "hidden_size", None)
        if not isinstance(hidden, int) or hidden < 1:
            raise RuntimeError("模型没有 hidden_size")
        self.dim = hidden

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            self.truncated += _truncation_count(self._tokenizer, batch, self.max_length)
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                cls = self._model(**encoded).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
            rows.append(np.asarray(normalized.cpu().numpy(), dtype=np.float64))
        return np.concatenate(rows, axis=0)


def _truncation_count(tokenizer: Any, texts: list[str], max_length: int) -> int:
    full = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return sum(1 for token_ids in full["input_ids"] if len(token_ids) > max_length)
