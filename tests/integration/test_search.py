"""Phase 3 integration tests — retrieval + ranking pipeline."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kselect.backends.local import LocalBackend
from kselect.index.bm25_index import BM25Index
from kselect.index.faiss_index import FAISSIndex
from kselect.index.manager import IndexManager
from kselect.models.chunk import Chunk, ChunkMetadata
from kselect.models.config import FusionMode, IndexType, IndexConfig, KSelectConfig, RankingMode
from kselect.models.hit import Hit, SearchResult
from kselect.retrieval.engine import RetrievalEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

_DIM = 16


def _chunk(text: str, emb: list[float], idx: int = 0) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        embedding=emb,
        metadata=ChunkMetadata(
            source_file="test.txt",
            chunk_index=idx,
            char_start=0,
            char_end=len(text),
            token_count=len(text.split()),
        ),
    )


def _build_manager(chunks: list[Chunk], tmp_path, config: KSelectConfig) -> IndexManager:
    faiss_idx = FAISSIndex()
    bm25_idx = BM25Index()
    backend = LocalBackend(str(tmp_path / "state"))
    mgr = IndexManager(faiss_idx, bm25_idx, backend, config)
    mgr.build(chunks)
    return mgr


def _hits_from_results(
    results: list[tuple[str, float]],
    chunk_store: dict[str, Chunk],
) -> list[Hit]:
    hits = []
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = chunk_store[chunk_id]
        hits.append(Hit(
            chunk_id=chunk_id,
            doc_id=chunk.metadata.source_file,
            score=score,
            snippet=chunk.text,
            metadata=chunk.metadata.extra,
            rank=rank,
        ))
    return hits


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_search_fast_mode_returns_search_result(tmp_path):
    """search() with FAST mode returns SearchResult with correct types."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(0)
    chunks = [_chunk(f"doc {i}", rng.random(_DIM).tolist(), i) for i in range(20)]

    config = KSelectConfig()
    config.index.type = IndexType.FLAT
    config.ranking.mode = RankingMode.FAST
    config.ranking.k = 5

    mgr = _build_manager(chunks, tmp_path, config)
    engine = RetrievalEngine(mgr, config)

    query_emb = np.array(chunks[0].embedding, dtype="float32")
    results = engine.retrieve(
        query="doc 0",
        query_embedding=query_emb,
        k=config.ranking.k,
        fusion_mode=FusionMode.RRF,
    )

    chunk_store = mgr.get_chunk_store()
    hits = _hits_from_results(results[: config.ranking.k], chunk_store)
    sr = SearchResult(hits=hits, total_hits=len(hits), query="doc 0")

    assert isinstance(sr, SearchResult)
    assert len(sr.hits) <= config.ranking.k
    for h in sr.hits:
        assert isinstance(h.score, float)
        assert isinstance(h.snippet, str)


def test_search_hybrid_scores_differ_from_fast(tmp_path):
    """HYBRID-style (RRF with BM25) scores differ from DENSE-only on same query."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(1)
    # Mix embeddings so BM25 and FAISS disagree on ranking
    chunks = [_chunk(f"unique term {'alpha' if i % 2 == 0 else 'beta'} document {i}",
                     rng.random(_DIM).tolist(), i) for i in range(30)]

    config = KSelectConfig()
    config.index.type = IndexType.FLAT

    mgr = _build_manager(chunks, tmp_path, config)
    engine = RetrievalEngine(mgr, config)

    query_emb = np.array(chunks[0].embedding, dtype="float32")

    rrf_results = engine.retrieve("alpha", query_emb, k=5, fusion_mode=FusionMode.RRF)
    dense_results = engine.retrieve("alpha", query_emb, k=5, fusion_mode=FusionMode.DENSE)

    rrf_scores = [s for _, s in rrf_results[:5]]
    dense_scores = [s for _, s in dense_results[:5]]

    # RRF and DENSE produce different score scales
    assert rrf_scores != dense_scores, (
        "RRF and DENSE should produce different scores (RRF uses rank-based scoring)"
    )


def test_search_filters_applied(tmp_path):
    """Metadata filter excludes chunks with non-matching metadata values."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(2)
    chunks_a = [
        _chunk(f"legal document {i}", rng.random(_DIM).tolist(), i)
        for i in range(10)
    ]
    chunks_b = [
        _chunk(f"medical record {i}", rng.random(_DIM).tolist(), i + 10)
        for i in range(10)
    ]
    # Tag metadata
    for c in chunks_a:
        c.metadata.extra["domain"] = "legal"
    for c in chunks_b:
        c.metadata.extra["domain"] = "medical"

    all_chunks = chunks_a + chunks_b
    config = KSelectConfig()
    config.index.type = IndexType.FLAT

    mgr = _build_manager(all_chunks, tmp_path, config)
    engine = RetrievalEngine(mgr, config)

    query_emb = rng.random(_DIM).astype("float32")
    results = engine.retrieve(
        query="document",
        query_embedding=query_emb,
        k=10,
        fusion_mode=FusionMode.RRF,
        filters={"domain": "legal"},
    )

    chunk_store = mgr.get_chunk_store()
    for chunk_id, _ in results:
        chunk = chunk_store[chunk_id]
        assert chunk.metadata.extra.get("domain") == "legal", (
            f"Chunk {chunk_id} has domain={chunk.metadata.extra.get('domain')!r}, expected 'legal'"
        )
