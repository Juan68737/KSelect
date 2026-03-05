"""
KSelect — public entry point for the RAG + vector search SDK.
Do not call __init__ directly. Use the from_* classmethods.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Any

from kselect.backends.local import LocalBackend
from kselect.context.assembler import ContextAssembler
from kselect.exceptions import (
    ConfigError as KSelectConfigError,
    IndexLoadError as KSelectVersionError,
    LLMError,
)
from kselect.index.bm25_index import BM25Index
from kselect.index.faiss_index import FAISSIndex
from kselect.index.manager import IndexManager
from kselect.models.answer import QueryResult, Source
from kselect.models.cache import CacheStats
from kselect.models.config import (
    ContextConfig,
    ContextStrategy,
    FusionMode,
    KSelectConfig,
    RankingMode,
)
from kselect.models.hit import Hit, SearchResult
from kselect.retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)

_RANKING_FLAGS = ("hybrid", "colbert", "fast", "cross", "none")
_FLAG_TO_MODE = {
    "hybrid": RankingMode.HYBRID,
    "colbert": RankingMode.COLBERT,
    "fast": RankingMode.FAST,
    "cross": RankingMode.CROSS,
    "none": RankingMode.NONE,
}


class KSelect:
    """
    Public entry point for KSelect. Do not call __init__ directly.
    Use classmethods: from_folder, from_csv, from_json, from_jsonl,
                      from_backend, from_yaml, load.
    """

    def __init__(
        self,
        config: KSelectConfig,
        _index_manager: IndexManager,
        _cache: "Any | None" = None,
        _llm: "Any | None" = None,
        _metrics: "Any | None" = None,
    ) -> None:
        self._config = config
        self._index_manager = _index_manager
        self._retrieval_engine = RetrievalEngine(_index_manager, config)
        self._assembler = ContextAssembler(llm=_llm)
        self._cache = _cache
        self._llm = _llm
        self._metrics = _metrics

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_folder(
        cls,
        path: str,
        *,
        config: KSelectConfig | None = None,
        index_type: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunking: str | None = None,
        hybrid_fcvi: bool | None = None,
        bm25: bool | None = None,
        metadata_fields: list[str] | None = None,
        batch_size: int | None = None,
        extract_tables: bool | None = None,
        max_docs: int | None = None,
        max_memory_gb: float | None = None,
    ) -> "KSelect":
        cfg = _clone_config(config)
        _apply_kwargs(cfg, dict(
            index_type=index_type,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking=chunking,
            hybrid_fcvi=hybrid_fcvi,
            bm25=bm25,
            batch_size=batch_size,
            extract_tables=extract_tables,
        ))

        state_dir = tempfile.mkdtemp(prefix="kselect_state_")
        backend = LocalBackend(state_dir)
        mgr = _build_manager(backend, cfg)

        from kselect.ingestion.loaders import FolderLoader
        from kselect.ingestion.pipeline import IngestionPipeline

        loader = FolderLoader(
            path,
            extract_tables=cfg.chunking.extract_tables,
            max_docs=max_docs,
        )
        pipeline = IngestionPipeline()
        chunks = pipeline.run(loader, cfg)
        mgr.build(chunks)

        return cls(cfg, mgr)

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        text_col: str,
        metadata: list[str] | None = None,
        vector_col: str | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect":
        cfg = _clone_config(config)
        _apply_kwargs(cfg, kwargs)

        from kselect.ingestion.loaders import CSVLoader
        from kselect.ingestion.pipeline import IngestionPipeline

        loader = CSVLoader(path, text_col=text_col, metadata=metadata or [])
        pipeline = IngestionPipeline()

        state_dir = tempfile.mkdtemp(prefix="kselect_state_")
        backend = LocalBackend(state_dir)
        mgr = _build_manager(backend, cfg)

        chunks = pipeline.run(loader, cfg)
        mgr.build(chunks)
        return cls(cfg, mgr)

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        text_key: str,
        metadata: list[str] | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect":
        cfg = _clone_config(config)
        _apply_kwargs(cfg, kwargs)

        from kselect.ingestion.loaders import JSONLoader
        from kselect.ingestion.pipeline import IngestionPipeline

        loader = JSONLoader(path, text_key=text_key, metadata_keys=metadata or [])
        pipeline = IngestionPipeline()

        state_dir = tempfile.mkdtemp(prefix="kselect_state_")
        backend = LocalBackend(state_dir)
        mgr = _build_manager(backend, cfg)

        chunks = pipeline.run(loader, cfg)
        mgr.build(chunks)
        return cls(cfg, mgr)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        *,
        text_key: str,
        metadata: list[str] | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect":
        return cls.from_json(path, text_key=text_key, metadata=metadata, config=config, **kwargs)

    @classmethod
    def from_backend(
        cls,
        uri: str,
        *,
        text_col: str = "content",
        metadata_cols: list[str] | None = None,
        config: KSelectConfig | None = None,
        **kwargs,
    ) -> "KSelect":
        cfg = _clone_config(config)
        _apply_kwargs(cfg, kwargs)

        from kselect.backends.factory import parse_backend_uri

        backend = parse_backend_uri(uri)
        mgr = _build_manager(backend, cfg)

        chunks = backend.get_all_chunks()
        logged = 0
        batch: list = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= 100_000:
                logged += len(batch)
                logger.info("from_backend: loaded %d chunks...", logged)
                batch = []

        all_chunks = list(backend.get_all_chunks())
        mgr.build(all_chunks)
        return cls(cfg, mgr)

    @classmethod
    def from_yaml(cls, path: str) -> "KSelect":
        cfg = KSelectConfig.from_yaml(path)
        if cfg.backend is None:
            raise KSelectConfigError(
                "from_yaml requires backend.uri to be set. "
                "For local folders, use from_folder() directly."
            )
        return cls.from_backend(cfg.backend.uri, config=cfg)

    @classmethod
    def load(cls, path: str) -> "KSelect":
        base = Path(path)
        version_file = base / "version.txt"
        if not version_file.exists():
            raise KSelectVersionError(f"version.txt missing in {path} — state may be corrupted.")

        backend = LocalBackend(path)
        faiss_idx = FAISSIndex()
        bm25_idx = BM25Index()
        # Load config from saved state
        import json
        raw_config = json.loads((base / "config.json").read_text())
        cfg = KSelectConfig.model_validate(raw_config)

        mgr = IndexManager(faiss_idx, bm25_idx, backend, cfg)
        mgr.load(path)

        cache = None
        cache_dir = base / "cache"
        if cache_dir.exists() and cfg.cache.enabled:
            try:
                from kselect.cache.semantic_cache import SemanticCache
                cache = SemanticCache(cfg.cache)
                # TODO: restore FAISS cache entries from cache_dir/entries.jsonl
            except Exception as exc:
                logger.warning("KSelect.load: failed to restore cache: %s", exc)

        return cls(cfg, mgr, _cache=cache)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        self._index_manager.save(path)

    # ── Incremental ingestion ─────────────────────────────────────────────────

    def add_doc(self, path: str, metadata: dict[str, Any] | None = None) -> int:
        from kselect.ingestion.loaders import FolderLoader
        from kselect.ingestion.pipeline import IngestionPipeline

        existing_ids = set(self._index_manager.get_chunk_store().keys())
        loader = FolderLoader(
            os.path.dirname(path) or ".",
            extract_tables=self._config.chunking.extract_tables,
            include_files=[os.path.basename(path)],
        )
        pipeline = IngestionPipeline()
        chunks = pipeline.run(loader, self._config, existing_chunk_ids=existing_ids)

        if metadata:
            for chunk in chunks:
                chunk.metadata.extra.update(metadata)

        if not chunks:
            logger.warning("KSelect.add_doc: no new chunks from %s", path)
            return 0

        self._index_manager.add_chunks(chunks)
        return len(chunks)

    def add_folder(self, path: str, metadata: dict[str, Any] | None = None) -> int:
        from kselect.ingestion.loaders import FolderLoader
        from kselect.ingestion.pipeline import IngestionPipeline

        existing_ids = set(self._index_manager.get_chunk_store().keys())
        loader = FolderLoader(path, extract_tables=self._config.chunking.extract_tables)
        pipeline = IngestionPipeline()
        chunks = pipeline.run(loader, self._config, existing_chunk_ids=existing_ids)

        if metadata:
            for chunk in chunks:
                chunk.metadata.extra.update(metadata)

        if not chunks:
            logger.warning("KSelect.add_folder: no new chunks from %s", path)
            return 0

        self._index_manager.add_chunks(chunks)
        return len(chunks)

    # ── Search (no LLM) ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        hybrid: bool = False,
        colbert: bool = False,
        fast: bool = False,
        cross: bool = False,
        none: bool = False,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        trace: bool | None = None,
    ) -> SearchResult:
        t0 = time.perf_counter()
        ranking_mode = _resolve_ranking_mode(hybrid, colbert, fast, cross, none, self._config)
        fusion_mode = _resolve_fusion_mode(fusion, self._config)
        effective_k = k if k is not None else self._config.ranking.k
        do_trace = trace if trace is not None else self._config.observability.trace

        # Embed query
        query_embedding = self._retrieval_engine.embed_query(query)

        # Cache check
        if self._cache is not None and self._config.cache.enabled:
            cached = self._cache.get(query_embedding, query, self._config.cache)
            if cached is not None and isinstance(cached, SearchResult):
                return cached

        # Retrieve
        candidates = self._retrieval_engine.retrieve(
            query=query,
            query_embedding=query_embedding,
            k=effective_k,
            fusion_mode=fusion_mode,
            filters=filters,
        )

        # Rank
        hits = _rank_candidates(
            candidates, ranking_mode, effective_k,
            query=query,
            query_embedding=query_embedding,
            chunk_store=self._index_manager.get_chunk_store(),
            config=self._config,
        )

        result = SearchResult(
            hits=hits,
            total_hits=len(hits),
            query=query,
        )

        # Cache store
        if self._cache is not None and self._config.cache.enabled:
            self._cache.set(query_embedding, query, result)

        if self._config.observability.log_queries:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            from kselect.observability.logging import log_search_event, get_logger
            log_search_event(
                get_logger(__name__),
                query=query,
                mode=str(ranking_mode),
                fusion=str(fusion_mode),
                cache_hit=False,
                latency_ms=latency_ms,
                chunks_retrieved=len(candidates),
                index_drift=self._index_manager.index_drift(),
            )

        return result

    async def search_async(self, query: str, **kwargs) -> SearchResult:
        import anyio  # type: ignore[import-untyped]
        return await anyio.to_thread.run_sync(lambda: self.search(query, **kwargs))

    # ── Query (search + LLM) ──────────────────────────────────────────────────

    def query(
        self,
        query: str,
        *,
        k: int | None = None,
        hybrid: bool = False,
        colbert: bool = False,
        fast: bool = False,
        cross: bool = False,
        none: bool = False,
        fusion: str | None = None,
        filters: dict[str, Any] | None = None,
        context_strategy: str | None = None,
        max_context_tokens: int | None = None,
        return_context: bool = False,
        model: str | None = None,
        trace: bool | None = None,
    ) -> QueryResult:
        if self._llm is None:
            raise LLMError(
                "KSelect.query() requires an LLMClient. "
                "Pass _llm= to the constructor or use KSelect with an LLM configured."
            )

        t0 = time.perf_counter()
        ranking_mode = _resolve_ranking_mode(hybrid, colbert, fast, cross, none, self._config)
        fusion_mode = _resolve_fusion_mode(fusion, self._config)
        effective_k = k if k is not None else self._config.ranking.k

        # Context config overrides
        ctx_cfg = self._config.context.model_copy()
        if context_strategy is not None:
            ctx_cfg.strategy = ContextStrategy(context_strategy)
        if max_context_tokens is not None:
            ctx_cfg.max_context_tokens = max_context_tokens

        # Embed query
        query_embedding = self._retrieval_engine.embed_query(query)

        # Cache check
        if self._cache is not None and self._config.cache.enabled:
            cached = self._cache.get(query_embedding, query, self._config.cache)
            if cached is not None and isinstance(cached, QueryResult):
                return cached

        # Retrieve + rank via search internals (reuse embedding)
        t_retrieval = time.perf_counter()
        candidates = self._retrieval_engine.retrieve(
            query=query,
            query_embedding=query_embedding,
            k=effective_k,
            fusion_mode=fusion_mode,
            filters=filters,
        )
        hits = _rank_candidates(
            candidates, ranking_mode, effective_k,
            query=query,
            query_embedding=query_embedding,
            chunk_store=self._index_manager.get_chunk_store(),
            config=self._config,
        )

        # Assemble context
        context_hits, context_tokens = self._assembler.assemble(hits, ctx_cfg)
        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

        # LLM call (synchronous wrapper around async generate)
        t_llm = time.perf_counter()
        answer, confidence = asyncio.run(
            self._llm.generate(query, context_hits, max_tokens=self._config.llm.max_tokens)
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000
        logger.debug(
            "query latency — retrieval: %.1fms | llm: %.1fms | total: %.1fms",
            retrieval_ms, llm_ms, (time.perf_counter() - t0) * 1000,
        )

        chunk_store = self._index_manager.get_chunk_store()
        sources = [
            Source(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                snippet=h.snippet,
                score=h.score,
                metadata=h.metadata,
            )
            for h in context_hits
        ]

        result = QueryResult(
            answer=answer,
            confidence=confidence,
            sources=sources,
            query=query,
            chunks_retrieved=len(candidates),
            chunks_in_context=len(context_hits),
            chunks_dropped=len(hits) - len(context_hits),
            context_tokens=context_tokens,
            max_context_tokens=ctx_cfg.max_context_tokens,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
        )

        if self._cache is not None and self._config.cache.enabled:
            self._cache.set(query_embedding, query, result)

        if self._config.observability.log_queries:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            from kselect.observability.logging import log_query_event, get_logger
            log_query_event(
                get_logger(__name__),
                query=query,
                mode=str(ranking_mode),
                fusion=str(fusion_mode),
                cache_hit=False,
                latency_ms=latency_ms,
                confidence=confidence,
                chunks_retrieved=len(candidates),
                chunks_in_context=len(context_hits),
                index_drift=self._index_manager.index_drift(),
            )

        return result

    async def query_async(self, query: str, **kwargs) -> QueryResult:
        # query() uses asyncio.run internally for LLM; run full query in thread
        import anyio  # type: ignore[import-untyped]
        return await anyio.to_thread.run_sync(lambda: self.query(query, **kwargs))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def index_size(self) -> int:
        return self._index_manager.index_size()

    def index_drift(self) -> float:
        return self._index_manager.index_drift()

    def recall_estimate(self) -> float:
        return self._index_manager.recall_estimate()

    def cache_stats(self) -> CacheStats:
        if self._cache is None:
            return CacheStats(
                hit_rate=0.0,
                false_positive_rate=0.0,
                size=0,
                llm_calls_saved=0,
                cost_saved_usd=0.0,
                qps_saved=0.0,
            )
        return self._cache.stats()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataset: list[dict],
        strategies: list[dict] | None = None,
    ) -> list:
        from kselect.eval.metrics import EvalMetrics, evaluate_dataset
        return evaluate_dataset(self, dataset, strategies)


# ── Module-level helpers ───────────────────────────────────────────────────────


def _clone_config(config: KSelectConfig | None) -> KSelectConfig:
    if config is None:
        return KSelectConfig()
    return config.model_copy(deep=True)


def _apply_kwargs(cfg: KSelectConfig, kwargs: dict) -> None:
    if kwargs.get("index_type") is not None:
        from kselect.models.config import IndexType
        cfg.index.type = IndexType(kwargs["index_type"])
    if kwargs.get("embedding_model") is not None:
        cfg.embedding.model = kwargs["embedding_model"]
    if kwargs.get("chunk_size") is not None:
        cfg.chunking.chunk_size = kwargs["chunk_size"]
    if kwargs.get("chunk_overlap") is not None:
        cfg.chunking.chunk_overlap = kwargs["chunk_overlap"]
    if kwargs.get("chunking") is not None:
        from kselect.models.config import ChunkingStrategy
        cfg.chunking.strategy = ChunkingStrategy(kwargs["chunking"])
    if kwargs.get("hybrid_fcvi") is not None:
        cfg.index.hybrid_fcvi = kwargs["hybrid_fcvi"]
    if kwargs.get("bm25") is not None:
        cfg.bm25.enabled = kwargs["bm25"]
    if kwargs.get("batch_size") is not None:
        cfg.embedding.batch_size = kwargs["batch_size"]
    if kwargs.get("extract_tables") is not None:
        cfg.chunking.extract_tables = kwargs["extract_tables"]


def _build_manager(backend: Any, cfg: KSelectConfig) -> IndexManager:
    return IndexManager(FAISSIndex(), BM25Index(), backend, cfg)


def _resolve_ranking_mode(
    hybrid: bool,
    colbert: bool,
    fast: bool,
    cross: bool,
    none: bool,
    config: KSelectConfig,
) -> RankingMode:
    flags = {"hybrid": hybrid, "colbert": colbert, "fast": fast, "cross": cross, "none": none}
    active = [k for k, v in flags.items() if v]
    if len(active) > 1:
        raise KSelectConfigError(
            f"Multiple ranking mode flags set: {active}. Set exactly one."
        )
    if len(active) == 1:
        return _FLAG_TO_MODE[active[0]]
    return RankingMode(config.ranking.mode)


def _resolve_fusion_mode(fusion: str | None, config: KSelectConfig) -> FusionMode:
    if fusion is not None:
        return FusionMode(fusion)
    return FusionMode(config.fusion.mode)


def _rank_candidates(
    candidates: list[tuple[str, float]],
    ranking_mode: RankingMode,
    k: int,
    *,
    query: str,
    query_embedding: Any,
    chunk_store: dict,
    config: KSelectConfig,
) -> list[Hit]:
    """Apply reranking strategy and return list of Hit objects."""
    mode_str = str(ranking_mode)

    if mode_str == RankingMode.HYBRID:
        # Cross-encoder rerank + MMR
        top_candidates = candidates[:k * 3]
        try:
            from kselect.ranking.cross_encoder import CrossEncoderReranker
            reranker = CrossEncoderReranker(config.ranking.cross_encoder_model)
            top_candidates = reranker.rerank(query, top_candidates, chunk_store, top_k=k * 2)
        except Exception as exc:
            logger.warning("Hybrid rerank (cross-encoder) failed: %s; using RRF scores.", exc)

        # MMR diversification
        embeddings = {}
        for chunk_id, _ in top_candidates:
            chunk = chunk_store.get(chunk_id)
            if chunk and chunk.embedding:
                import numpy as np
                embeddings[chunk_id] = np.array(chunk.embedding, dtype="float32")
        if embeddings:
            from kselect.ranking.mmr import mmr_diversify
            top_candidates = mmr_diversify(
                top_candidates, embeddings, top_k=k, lambda_=config.ranking.mmr_lambda
            )
        else:
            top_candidates = top_candidates[:k]

    elif mode_str == RankingMode.COLBERT:
        try:
            from kselect.ranking.colbert import ColBERTReranker
            reranker = ColBERTReranker(config.ranking.colbert_model)
            candidates = reranker.rerank(query, candidates, chunk_store, top_k=k)
        except Exception as exc:
            logger.warning("ColBERT rerank failed: %s; using retrieval scores.", exc)
            candidates = candidates[:k]
        top_candidates = candidates

    elif mode_str == RankingMode.CROSS:
        try:
            from kselect.ranking.cross_encoder import CrossEncoderReranker
            reranker = CrossEncoderReranker(config.ranking.cross_encoder_model)
            candidates = reranker.rerank(query, candidates, chunk_store, top_k=k)
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed: %s; using retrieval scores.", exc)
            candidates = candidates[:k]
        top_candidates = candidates

    else:
        # FAST and NONE: no reranking
        top_candidates = candidates[:k]

    # Build Hit objects
    hits: list[Hit] = []
    for rank, (chunk_id, score) in enumerate(top_candidates, start=1):
        chunk = chunk_store.get(chunk_id)
        if chunk is None:
            continue
        hits.append(Hit(
            chunk_id=chunk_id,
            doc_id=chunk.metadata.source_file,
            score=score,
            snippet=chunk.text,
            metadata=chunk.metadata.extra,
            rank=rank,
        ))
    return hits
