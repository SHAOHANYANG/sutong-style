"""The model server, its clients, and the real service assembly. CPU only, fakes for models."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from agent.prompts import SYSTEM_LEAD
from api import deps
from api.assembly import build_services
from api.main import app
from api.settings import ServiceSettings, service_agent_config
from infra.model_server_client import ChatCompletionsGenerator, RemoteEmbedder
from retrieval.dense import save_dense_cache
from retrieval.prompt import build_prompt
from scripts.build_dense_index import train_documents
from scripts.serve_model import create_app
from scripts.split_corpus import CorpusSplit
from scripts.train import Pair
from tests.fakes import FakeAgentGenerator, FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]
REVISION = "rev-test"


class _FakeChat:
    def __init__(self, tokens: int = 10) -> None:
        self.tokens = tokens
        self.calls: list[list[dict[str, str]]] = []

    def count(self, messages: list[dict[str, str]]) -> int:
        return self.tokens

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        self.calls.append(messages)
        return "假输出：" + messages[-1]["content"]


def _server(chat: _FakeChat | None = None, *, with_embedder: bool = True) -> TestClient:
    return TestClient(
        create_app(
            chat=chat or _FakeChat(),
            chat_name="sutong-v2",
            embedder=FakeEmbedder(dim=4) if with_embedder else None,
            embedding_name="BAAI/bge-m3",
            embedding_revision=REVISION,
        )
    )


def test_model_server_speaks_the_openai_shapes() -> None:
    chat = _FakeChat()
    client = _server(chat)
    listed = client.get("/v1/models").json()["data"]
    assert [item["id"] for item in listed] == ["sutong-v2", "BAAI/bge-m3"]
    assert listed[1]["revision"] == REVISION
    messages = [{"role": "system", "content": "指令"}, {"role": "user", "content": "甲去井边"}]
    done = client.post("/v1/chat/completions", json={"model": "sutong-v2", "messages": messages})
    assert done.status_code == 200
    assert done.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "假输出：甲去井边",
    }
    assert chat.calls == [messages]
    vectors = client.post("/v1/embeddings", json={"model": "BAAI/bge-m3", "input": ["甲", "乙"]})
    assert [len(item["embedding"]) for item in vectors.json()["data"]] == [4, 4]


def test_model_server_rejects_what_it_cannot_do_faithfully() -> None:
    body = {"model": "sutong-v2", "messages": [{"role": "user", "content": "甲"}]}
    client = _server(_FakeChat(tokens=4000))
    # 4000 + 768 does not fit 4096: an error, never a truncated prompt.
    over = client.post("/v1/chat/completions", json=body)
    assert over.status_code == 400
    assert "max_seq_length" in over.json()["detail"]
    client = _server()
    assert client.post("/v1/chat/completions", json={**body, "model": "x"}).status_code == 404
    assert client.post("/v1/chat/completions", json={**body, "stream": True}).status_code == 400
    assert client.post("/v1/chat/completions", json={**body, "temperature": 0.7}).status_code == 400
    assert client.post("/v1/chat/completions", json={**body, "max_tokens": 9999}).status_code == 422
    no_embedder = _server(with_embedder=False)
    missing = no_embedder.post("/v1/embeddings", json={"model": "BAAI/bge-m3", "input": ["甲"]})
    assert missing.status_code == 404


def test_clients_round_trip_through_the_server() -> None:
    chat = _FakeChat()
    http = _server(chat)
    generator = ChatCompletionsGenerator(
        base_url="http://testserver/v1", model="sutong-v2", http_client=http
    )
    messages = build_prompt("甲去井边", [])
    assert generator.generate(messages) == "假输出：甲去井边"
    assert generator.generate_with_seed(messages, seed=7) == "假输出：甲去井边"
    assert chat.calls == [messages, messages]
    assert generator.ping() is True
    other = ChatCompletionsGenerator(
        base_url="http://testserver/v1", model="absent", http_client=http
    )
    assert other.ping() is False
    embedder = RemoteEmbedder(
        base_url="http://testserver/v1",
        model="BAAI/bge-m3",
        expected_revision=REVISION,
        http_client=http,
    )
    np.testing.assert_allclose(
        embedder.encode(["甲", "乙"]), FakeEmbedder(dim=4).encode(["甲", "乙"])
    )
    assert embedder.ping() is True
    stale = RemoteEmbedder(
        base_url="http://testserver/v1",
        model="BAAI/bge-m3",
        expected_revision="another-revision",
        http_client=http,
    )
    # Vectors from a different encoder revision must not be compared with the index.
    assert stale.ping() is False


def _pair(doc_id: str, split: str, vernacular: str, original: str) -> Pair:
    return Pair(
        id=doc_id, work="自编", idx=1, vernacular=vernacular, original=original, split=split
    )


def _world(tmp_path: Path) -> ServiceSettings:
    pairs = [
        _pair("t1", "train", "甲去井边打水", "井边有一个人在打水"),
        _pair("t2", "train", "乙买了三斤米", "三斤米放在门口"),
        _pair("t3", "train", "丙抬头看着天", "天色慢慢暗下来"),
        _pair("t4", "train", "丁把门关上了", "门已经关上"),
        _pair("e1", "eval", "戊要出门", "门开着"),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(row.model_dump_json() + "\n" for row in pairs), encoding="utf-8")
    split = CorpusSplit(seed=42, eval_ratio=0.2, train=["t1", "t2", "t3", "t4"], eval=["e1"])
    split_path = tmp_path / "split.json"
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    documents = train_documents(pairs, split)
    index_dir = tmp_path / "index"
    save_dense_cache(
        documents,
        FakeEmbedder(dim=4).encode([document.text for document in documents]),
        model_id="BAAI/bge-m3",
        revision=REVISION,
        max_length=512,
        directory=index_dir,
    )
    plan = yaml.safe_load((ROOT / "eval/configs/retrieval.yaml").read_text(encoding="utf-8"))
    plan["embedding_revision"] = REVISION
    plan_path = tmp_path / "retrieval.yaml"
    plan_path.write_text(yaml.safe_dump(plan, allow_unicode=True), encoding="utf-8")
    return ServiceSettings(
        model_base_url="http://model.test/v1",
        pairs=pairs_path,
        split=split_path,
        index_dir=index_dir,
        style_config=ROOT / "eval/configs/eval59.yaml",
        retrieval_config=plan_path,
    )


def test_settings_come_from_prefixed_environment_variables() -> None:
    assert ServiceSettings.from_env({}).model_base_url is None
    settings = ServiceSettings.from_env(
        {
            "SUTONG_MODEL_BASE_URL": "http://gpu:8001/v1",
            "SUTONG_PAIRS": "x/pairs.jsonl",
            "PATH": "/",
        }
    )
    assert settings.model_base_url == "http://gpu:8001/v1"
    assert settings.pairs == Path("x/pairs.jsonl")
    assert settings.model_name == "sutong-v2"


def test_unconfigured_service_keeps_the_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUTONG_MODEL_BASE_URL", raising=False)
    deps.services.cache_clear()
    assert deps.services() is None
    with pytest.raises(NotImplementedError, match="SUTONG_MODEL_BASE_URL"):
        deps.get_generator().generate([])
    deps.services.cache_clear()


def test_assembly_retrieves_train_exemplars_in_rank_order(tmp_path: Path) -> None:
    settings = _world(tmp_path)
    built = build_services(settings, generator=FakeAgentGenerator(), embedder=FakeEmbedder(dim=4))
    exemplars = list(built.retriever.retrieve("乙买了三斤米"))
    # primary_k of the committed retrieval config, and only training rows.
    assert len(exemplars) == 2
    assert {item.id for item in exemplars} <= {"t1", "t2", "t3", "t4"}
    assert exemplars[0].id == "t2"
    assert exemplars[0].original == "三斤米放在门口"
    assert built.retriever.retrieve("乙买了三斤米") == exemplars
    assert built.verifier.verify("戊带着12块钱出门", "戊出门了")[0].kind == "numeral_missing"
    assert built.scorer.score("戊要出门", "门开着") <= 0.0
    with pytest.raises(ValueError, match="模型服务地址"):
        build_services(settings.model_copy(update={"model_base_url": None}))


def test_service_runs_the_assembled_pipeline_end_to_end(tmp_path: Path) -> None:
    settings = _world(tmp_path)
    generator = FakeAgentGenerator(["戊出门了", "戊带着十二块钱出门"])
    built = build_services(settings, generator=generator, embedder=FakeEmbedder(dim=4))
    app.dependency_overrides[deps.get_retriever] = lambda: built.retriever
    app.dependency_overrides[deps.get_generator] = lambda: built.generator
    app.dependency_overrides[deps.get_verifier] = lambda: built.verifier
    app.dependency_overrides[deps.get_scorer] = lambda: built.scorer

    async def post() -> dict[str, object]:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.post(
                "/v1/transform", json={"text": "戊带着12块钱出门", "stream": False}
            )
        assert response.status_code == 200
        return dict(json.loads(response.text))

    try:
        body = asyncio.run(post())
    finally:
        app.dependency_overrides.clear()
    assert body["output"] == "戊带着十二块钱出门"
    assert body["selected_round"] == 2
    assert body["termination"] == "accepted"
    first, second = generator.calls
    # Round one carries the two retrieved exemplars as chat turns.
    assert len(first) == 1 + 2 * 2 + 1
    # The service loop puts feedback in the system turn and leaves the user turns alone.
    assert service_agent_config().echo_guard is True
    assert SYSTEM_LEAD in second[0]["content"]
    assert "「12块」" in second[0]["content"]
    assert second[1:] == first[1:]
