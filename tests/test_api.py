"""CPU-only API and SSE integration tests."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from agent.state import AgentResult
from api.deps import (
    StyleConfig,
    get_generator,
    get_retriever,
    get_scorer,
    get_style_config,
    get_verifier,
)
from api.main import app
from eval.fidelity import Violation
from tests.fakes import (
    FakeAgentGenerator,
    FakeAgentRetriever,
    FakeAgentScorer,
    FakeAgentVerifier,
    FakeUnhealthyAgentGenerator,
)

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _dependencies(
    *,
    generator: FakeAgentGenerator | None = None,
    retriever: FakeAgentRetriever | None = None,
    verifier: FakeAgentVerifier | None = None,
    scorer: FakeAgentScorer | None = None,
) -> Iterator[None]:
    app.dependency_overrides[get_generator] = lambda: generator or FakeAgentGenerator()
    app.dependency_overrides[get_retriever] = lambda: retriever or FakeAgentRetriever()
    app.dependency_overrides[get_verifier] = lambda: verifier or FakeAgentVerifier()
    app.dependency_overrides[get_scorer] = lambda: scorer or FakeAgentScorer()
    app.dependency_overrides[get_style_config] = lambda: StyleConfig(adapters=["sutong"])
    try:
        yield
    finally:
        app.dependency_overrides.clear()


async def _post(payload: dict[str, object]) -> tuple[int, str, str]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/transform", json=payload)
    return response.status_code, response.headers.get("content-type", ""), response.text


async def _get(path: str) -> tuple[int, dict[str, object] | list[object]]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    return response.status_code, response.json()


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    parsed: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        parsed.append((event, json.loads(data)))
    return parsed


def test_stream_emits_tokens_traces_and_done_without_prose_in_trace() -> None:
    marker = "UNIQUE_INPUT_MARKER_9f8c"
    generator = FakeAgentGenerator([f"改写结果含有{marker}"])
    with _dependencies(generator=generator):
        status, content_type, body = asyncio.run(
            _post({"text": marker, "stream": True, "seed": 77})
        )

    events = _parse_sse(body)
    assert status == 200
    assert content_type.startswith("text/event-stream")
    assert {name for name, _ in events} >= {"token", "trace", "done"}
    traces = [data for name, data in events if name == "trace"]
    assert [trace["node"] for trace in traces] == [
        "retrieve",
        "generate",
        "verify",
        "score",
        "route",
    ]
    assert marker not in json.dumps(traces, ensure_ascii=False)
    assert marker in "".join(str(data["text"]) for name, data in events if name == "token")
    done = next(data for name, data in events if name == "done")
    assert marker in str(done["output"])
    assert done["seed"] == 77
    assert generator.seeds == [77]


def test_stream_revision_tokens_carry_both_rounds() -> None:
    violation = Violation(kind="title", expected="太医", actual="宫监")
    generator = FakeAgentGenerator(["第一版", "第二版修好"])
    verifier = FakeAgentVerifier(batches=[[violation], []])
    with _dependencies(generator=generator, verifier=verifier):
        _, _, body = asyncio.run(_post({"text": "太医进门", "stream": True}))

    events = _parse_sse(body)
    token_rounds = {data["round"] for name, data in events if name == "token"}
    done = next(data for name, data in events if name == "done")
    assert token_rounds == {1, 2}
    assert done["selected_round"] == 2


def test_non_stream_response_uses_agent_result_fields() -> None:
    with _dependencies(generator=FakeAgentGenerator(["成稿"])):
        status, content_type, body = asyncio.run(
            _post({"text": "白话", "stream": False, "max_iter": 1, "seed": 9})
        )

    data = json.loads(body)
    assert status == 200
    assert content_type.startswith("application/json")
    assert set(AgentResult.model_fields) <= set(data)
    assert data["output"] == "成稿"
    assert data["selected_round"] == 1
    assert data["total_rounds"] == 1
    assert data["termination"] == "accepted"
    assert data["fallback_kind"] is None
    assert data["seed"] == 9


def test_stream_generator_error_returns_labeled_fallback() -> None:
    generator = FakeAgentGenerator(errors_on=[0])
    with _dependencies(generator=generator):
        _, _, body = asyncio.run(_post({"text": "原输入", "stream": True}))

    events = _parse_sse(body)
    done = next(data for name, data in events if name == "done")
    assert done["output"] == "原输入"
    assert done["termination"] == "fallback"
    assert done["fallback_kind"] == "first_round_error"
    assert not any(name == "error" for name, _ in events)


def test_healthz_reports_dependency_failure_and_process_stays_responsive() -> None:
    with _dependencies(generator=FakeUnhealthyAgentGenerator()):
        first_status, first = asyncio.run(_get("/healthz"))
        second_status, second = asyncio.run(_get("/healthz"))

    assert first_status == second_status == 503
    assert first == second
    assert isinstance(first, dict)
    dependencies = first["dependencies"]
    assert isinstance(dependencies, dict)
    assert dependencies["generator"]["ok"] is False
    assert dependencies["retriever"]["ok"] is True


def test_healthz_ok() -> None:
    with _dependencies():
        status, body = asyncio.run(_get("/healthz"))
    assert status == 200
    assert isinstance(body, dict)
    assert body["status"] == "ok"


def test_request_validation() -> None:
    with _dependencies():
        empty = asyncio.run(_post({"text": "", "stream": False}))[0]
        whitespace = asyncio.run(_post({"text": "   ", "stream": False}))[0]
        zero = asyncio.run(_post({"text": "有效", "max_iter": 0, "stream": False}))[0]
        four = asyncio.run(_post({"text": "有效", "max_iter": 4, "stream": False}))[0]
    assert (empty, whitespace, zero, four) == (422, 422, 422, 422)


def test_styles_returns_injected_adapter_list() -> None:
    with _dependencies():
        status, body = asyncio.run(_get("/v1/styles"))
    assert status == 200
    assert body == ["sutong"]


def test_importing_api_does_not_import_gpu_stacks() -> None:
    script = (
        "import sys; import api.main; "
        "print([name for name in ('torch','transformers','unsloth') if name in sys.modules])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.strip() == "[]"
