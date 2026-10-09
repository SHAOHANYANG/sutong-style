"""Serve the fine-tuned model and the query encoder over an OpenAI-compatible HTTP API.

This is the GPU process. The API process (api/) talks to it with the openai SDK and
never imports torch. Decoding goes through scripts.generate.greedy_decode, the same
function every offline run used, so a served completion equals the offline one.

Three routes: GET /v1/models, POST /v1/chat/completions, POST /v1/embeddings.
Heavy imports live inside functions so --help works without a GPU stack.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.prompt import MAX_NEW_TOKENS, MAX_SEQ_LENGTH
from retrieval.types import Embedder

LOGGER = structlog.get_logger()
DEFAULT_PORT = 8001


class ChatModel(Protocol):
    """Greedy completion plus the prompt length the budget check needs."""

    def count(self, messages: list[dict[str, str]]) -> int: ...

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str: ...


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=MAX_NEW_TOKENS)
    seed: int | None = None
    temperature: float | None = None
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] = Field(min_length=1)


class UnslothChatModel:
    """The loaded adapter. torch and unsloth were imported by the loader."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        from infra.qwen_prompt_tokenizer import QwenPromptTokenizer

        self._model = model
        self._tokenizer = tokenizer
        self._counter = QwenPromptTokenizer(tokenizer)

    def count(self, messages: list[dict[str, str]]) -> int:
        return self._counter.count(messages)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        from scripts.generate import greedy_decode

        return greedy_decode(self._model, self._tokenizer, messages, max_new_tokens)


def create_app(
    *,
    chat: ChatModel,
    chat_name: str,
    embedder: Embedder | None,
    embedding_name: str,
    embedding_revision: str | None,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> FastAPI:
    """Wire the three routes around injected models. Tests pass fakes."""
    app = FastAPI(title="sutong-style model server", version="0.1.0")
    # One card: completions and embeddings take turns instead of competing for memory.
    gpu = threading.Lock()

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        data: list[dict[str, object]] = [{"id": chat_name, "object": "model"}]
        if embedder is not None:
            data.append({"id": embedding_name, "object": "model", "revision": embedding_revision})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, object]:
        if request.model != chat_name:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        if request.temperature not in (None, 0.0):
            raise HTTPException(status_code=400, detail="only greedy decoding is supported")
        messages = [item.model_dump() for item in request.messages]
        started = time.perf_counter()
        with gpu:
            prompt_tokens = chat.count(messages)
            # Same rule as the offline runs: an over-long prompt is an error, never truncated.
            if prompt_tokens + request.max_tokens > max_seq_length:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"prompt tokens {prompt_tokens} + max_tokens {request.max_tokens}"
                        f" exceed max_seq_length {max_seq_length}"
                    ),
                )
            text = chat.generate(messages, request.max_tokens)
        LOGGER.info(
            "chat_completion",
            prompt_tokens=prompt_tokens,
            output_chars=len(text),
            seconds=round(time.perf_counter() - started, 3),
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens},
        }

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, object]:
        if embedder is None or request.model != embedding_name:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        with gpu:
            matrix = embedder.encode(list(request.input))
        return {
            "object": "list",
            "model": embedding_name,
            "data": [
                {"object": "embedding", "index": index, "embedding": [float(x) for x in row]}
                for index, row in enumerate(matrix)
            ],
        }

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 GPU 上提供生成与查询向量的 OpenAI 兼容接口")
    parser.add_argument("--adapter", type=Path, default=Path("adapters/sutong-v2/adapter"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="默认有 CUDA 用 CUDA；显存不够时传 cpu",
    )
    parser.add_argument("--no-embeddings", action="store_true", help="不加载查询编码器")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    structlog.configure(processors=[structlog.processors.JSONRenderer(ensure_ascii=False)])
    args = parse_args(argv)
    if not args.adapter.exists():
        raise SystemExit(f"adapter 不存在: {args.adapter}")
    import uvicorn

    from scripts.generate import load_inference_model, model_name

    model, tokenizer = load_inference_model(args.adapter, args.max_seq_length, args.seed)
    embedder: Embedder | None = None
    revision: str | None = None
    if not args.no_embeddings:
        from infra.bge_embedder import BgeEmbedder

        loaded = BgeEmbedder(
            args.embedding_model,
            revision=args.embedding_revision,
            max_length=args.embedding_max_length,
            device=args.embedding_device,
        )
        embedder, revision = loaded, loaded.revision
    app = create_app(
        chat=UnslothChatModel(model, tokenizer),
        chat_name=model_name(args.adapter),
        embedder=embedder,
        embedding_name=args.embedding_model,
        embedding_revision=revision,
        max_seq_length=args.max_seq_length,
    )
    LOGGER.info("model_server_start", host=args.host, port=args.port, embeddings=revision)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
