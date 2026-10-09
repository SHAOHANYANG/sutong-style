"""Build the real pipeline for the service: retrieval, generation, verification, scoring.

Imported only when a model server is configured. Everything heavy stays in that
server; this process reads the corpus and the committed indexes and never imports torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from agent.nodes import Generator, Retriever, Scorer, Verifier
from agent.scorer import PredictorScorer
from agent.verifier import FidelityVerifier
from api.settings import ServiceSettings
from eval.loaders import load_config, load_gazetteer, load_style_reference
from infra.model_server_client import ChatCompletionsGenerator, RemoteEmbedder
from retrieval.bm25 import Bm25Index
from retrieval.dense import DenseIndex, load_dense_cache
from retrieval.hybrid import HybridRetriever
from retrieval.prompt import Exemplar
from retrieval.style_index import StyleIndex
from retrieval.style_predictor import fit_predictor
from retrieval.types import Embedder
from scripts.build_dense_index import train_documents
from scripts.build_retrieval_plan import load_plan_config
from scripts.split_corpus import CorpusSplit
from scripts.train import Pair, load_pairs, select

LOGGER = structlog.get_logger()


@dataclass(frozen=True)
class Services:
    retriever: Retriever
    generator: Generator
    verifier: Verifier
    scorer: Scorer


class HybridExemplarRetriever:
    """Agent Retriever over the three-route index. Exemplars come back in rank order."""

    def __init__(
        self,
        hybrid: HybridRetriever,
        by_id: Mapping[str, Pair],
        k: int,
        health: RemoteEmbedder | None = None,
    ) -> None:
        self._hybrid = hybrid
        self._by_id = by_id
        self._k = k
        self._health = health

    def retrieve(self, vernacular: str) -> list[Exemplar]:
        return [
            Exemplar(
                id=hit.id,
                vernacular=self._by_id[hit.id].vernacular,
                original=self._by_id[hit.id].original,
            )
            for hit in self._hybrid.search(vernacular, self._k)
        ]

    def ping(self) -> bool:
        return True if self._health is None else self._health.ping()


def build_services(
    settings: ServiceSettings,
    *,
    generator: Generator | None = None,
    embedder: Embedder | None = None,
) -> Services:
    """Assemble the primary configuration of SPEC 4.6. Tests inject the two remote parts."""
    if settings.model_base_url is None and (generator is None or embedder is None):
        raise ValueError("没有配置模型服务地址")
    plan = load_plan_config(settings.retrieval_config)
    style_config = load_config(settings.style_config)
    reference = load_style_reference(style_config)
    pairs = load_pairs(settings.pairs)
    split = CorpusSplit.model_validate_json(settings.split.read_text(encoding="utf-8"))
    documents = train_documents(pairs, split)
    train = select(pairs, "train")
    if not documents:
        raise ValueError("训练集为空，无法装配检索")
    remote: RemoteEmbedder | None = None
    if embedder is None:
        remote = RemoteEmbedder(
            base_url=str(settings.model_base_url),
            model=plan.embedding_model,
            expected_revision=plan.embedding_revision,
        )
        embedder = remote
    if generator is None:
        generator = ChatCompletionsGenerator(
            base_url=str(settings.model_base_url), model=settings.model_name
        )
    matrix = load_dense_cache(
        documents,
        model_id=plan.embedding_model,
        revision=plan.embedding_revision,
        max_length=plan.max_length,
        directory=settings.index_dir,
    )
    original_z = np.vstack([reference.zscore(row.original) for row in train])
    predictor = fit_predictor(
        np.vstack([reference.zscore(row.vernacular) for row in train]),
        original_z,
        [row.work for row in train],
    )
    weights = plan.configs[plan.primary_config]
    hybrid = HybridRetriever(
        Bm25Index(documents),
        DenseIndex([document.id for document in documents], matrix, embedder),
        StyleIndex([row.id for row in train], original_z),
        reference,
        predictor,
        depth=plan.depth,
        rrf_k=plan.rrf_k,
        weights={"bm25": weights.bm25, "dense": weights.dense, "style": weights.style},
    )
    LOGGER.info(
        "service_assembled",
        documents=len(documents),
        fusion=plan.primary_config,
        k=plan.primary_k,
        model=settings.model_name,
    )
    return Services(
        retriever=HybridExemplarRetriever(
            hybrid, {row.id: row for row in train}, plan.primary_k, remote
        ),
        generator=generator,
        verifier=FidelityVerifier(load_gazetteer(Path(style_config.gazetteer))),
        scorer=PredictorScorer(reference, predictor),
    )
