"""HTTP routes and the synchronous-graph to asynchronous-SSE bridge."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Annotated, Protocol, cast, runtime_checkable

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.errors import GraphRecursionError

from agent.config import AgentConfig
from agent.graph import build_graph, build_result, initial_state, run_agent
from agent.nodes import Generator, Retriever, Scorer, Verifier
from agent.prompts import make_revision_message_builder
from agent.state import RECURSION_LIMIT, AgentResult, AgentState
from api.deps import (
    StyleConfig,
    get_agent_config,
    get_generator,
    get_retriever,
    get_scorer,
    get_style_config,
    get_verifier,
)
from api.schemas import (
    DependencyStatus,
    ErrorEvent,
    HealthResponse,
    TokenEvent,
    TransformRequest,
    TransformResponse,
)

TOKEN_CHUNK_CHARS = 24

router = APIRouter()


@runtime_checkable
class _SeedAwareGenerator(Protocol):
    def generate_with_seed(self, messages: list[dict[str, str]], *, seed: int) -> str: ...


@runtime_checkable
class _HealthCheck(Protocol):
    def ping(self) -> bool: ...


class _RequestGenerator:
    """Bind one request seed without changing the agent Generator protocol."""

    def __init__(self, generator: Generator, seed: int) -> None:
        self._generator = generator
        self._seed = seed

    def generate(self, messages: list[dict[str, str]]) -> str:
        if isinstance(self._generator, _SeedAwareGenerator):
            return self._generator.generate_with_seed(messages, seed=self._seed)
        return self._generator.generate(messages)


def _chunks(text: str) -> Iterator[str]:
    for start in range(0, len(text), TOKEN_CHUNK_CHARS):
        yield text[start : start + TOKEN_CHUNK_CHARS]


def _run_streaming_graph(
    request: TransformRequest,
    *,
    retriever: Retriever,
    generator: Generator,
    verifier: Verifier,
    scorer: Scorer,
    base_config: AgentConfig,
    emit: Callable[[str, str], None],
) -> AgentResult:
    """Run LangGraph synchronously, emitting each completed node immediately."""
    settings = base_config.model_copy(update={"max_generations": request.max_iter})
    compiled, box = build_graph(
        retriever=retriever,
        generator=_RequestGenerator(generator, request.seed),
        verifier=verifier,
        scorer=scorer,
        message_builder=make_revision_message_builder(settings.feedback_format),
    )
    initial = initial_state(request.text, settings)
    state = initial
    emitted_trace_count = 0
    try:
        values = compiled.stream(
            initial,
            {"recursion_limit": RECURSION_LIMIT},
            stream_mode="values",
        )
        for raw_state in values:
            state = cast(AgentState, raw_state)
            trace = list(state.get("trace") or [])
            for event in trace[emitted_trace_count:]:
                emit("trace", event.model_dump_json())
                if event.node == "generate":
                    round_number = int(event.payload.get("round") or 0)
                    candidates = list(state.get("candidates") or [])
                    if 0 < round_number <= len(candidates):
                        for text in _chunks(candidates[round_number - 1]):
                            emit(
                                "token",
                                TokenEvent(round=round_number, text=text).model_dump_json(),
                            )
            emitted_trace_count = len(trace)
    except GraphRecursionError:
        state = cast(AgentState, dict(box.state or state))
        state["hit_recursion_limit"] = True
    return build_result(state, request.text)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _event_stream(
    request: TransformRequest,
    *,
    retriever: Retriever,
    generator: Generator,
    verifier: Verifier,
    scorer: Scorer,
    base_config: AgentConfig,
) -> AsyncIterator[str]:
    events: queue.Queue[tuple[str, str] | None] = queue.Queue()

    def emit(event: str, data: str) -> None:
        events.put((event, data))

    def produce() -> None:
        try:
            result = _run_streaming_graph(
                request,
                retriever=retriever,
                generator=generator,
                verifier=verifier,
                scorer=scorer,
                base_config=base_config,
                emit=emit,
            )
            response = TransformResponse(**result.model_dump(), seed=request.seed)
            emit("done", response.model_dump_json())
        except Exception as exc:
            error = ErrorEvent(
                error_type=type(exc).__name__,
                detail="streaming transform failed",
            )
            emit("error", error.model_dump_json())
        finally:
            events.put(None)

    worker = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while True:
            item = await asyncio.to_thread(events.get)
            if item is None:
                break
            yield _sse(*item)
    finally:
        await worker


def _run_non_streaming(
    request: TransformRequest,
    retriever: Retriever,
    generator: Generator,
    verifier: Verifier,
    scorer: Scorer,
    base_config: AgentConfig,
) -> TransformResponse:
    result = run_agent(
        request.text,
        retriever=retriever,
        generator=_RequestGenerator(generator, request.seed),
        verifier=verifier,
        scorer=scorer,
        config=base_config.model_copy(update={"max_generations": request.max_iter}),
    )
    return TransformResponse(**result.model_dump(), seed=request.seed)


@router.post("/v1/transform", response_model=None)
async def transform(
    request: TransformRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    generator: Annotated[Generator, Depends(get_generator)],
    verifier: Annotated[Verifier, Depends(get_verifier)],
    scorer: Annotated[Scorer, Depends(get_scorer)],
    base_config: Annotated[AgentConfig, Depends(get_agent_config)],
) -> TransformResponse | StreamingResponse:
    if not request.stream:
        return await asyncio.to_thread(
            _run_non_streaming,
            request,
            retriever,
            generator,
            verifier,
            scorer,
            base_config,
        )
    return StreamingResponse(
        _event_stream(
            request,
            retriever=retriever,
            generator=generator,
            verifier=verifier,
            scorer=scorer,
            base_config=base_config,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/v1/styles", response_model=list[str])
def styles(config: Annotated[StyleConfig, Depends(get_style_config)]) -> list[str]:
    return list(config.adapters)


def _probe(dependency: object) -> DependencyStatus:
    if not isinstance(dependency, _HealthCheck):
        return DependencyStatus(ok=True, detail="no ping hook; dependency is injected")
    try:
        if dependency.ping():
            return DependencyStatus(ok=True)
        return DependencyStatus(ok=False, detail="ping returned false")
    except Exception as exc:
        return DependencyStatus(ok=False, detail=f"ping raised {type(exc).__name__}")


@router.get("/healthz", response_model=None)
def healthz(
    generator: Annotated[Generator, Depends(get_generator)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> HealthResponse | JSONResponse:
    dependencies = {
        "generator": _probe(generator),
        "retriever": _probe(retriever),
    }
    healthy = all(status.ok for status in dependencies.values())
    response = HealthResponse(
        status="ok" if healthy else "unhealthy",
        dependencies=dependencies,
    )
    if healthy:
        return response
    return JSONResponse(status_code=503, content=response.model_dump())
