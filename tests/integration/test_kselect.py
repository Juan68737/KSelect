"""Phase 5-6 integration tests — KSelect.search() and KSelect.query()."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from kselect.backends.local import LocalBackend
from kselect.index.bm25_index import BM25Index
from kselect.index.faiss_index import FAISSIndex
from kselect.index.manager import IndexManager
from kselect.kselect import KSelect
from kselect.models.answer import QueryResult
from kselect.models.chunk import Chunk, ChunkMetadata
from kselect.models.config import IndexType, KSelectConfig
from kselect.models.hit import SearchResult

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


def _build_ks(
    chunks: list[Chunk],
    tmp_path,
    *,
    llm=None,
    cache=None,
) -> KSelect:
    cfg = KSelectConfig()
    cfg.index.type = IndexType.FLAT
    backend = LocalBackend(str(tmp_path / "state"))
    mgr = IndexManager(FAISSIndex(), BM25Index(), backend, cfg)
    mgr.build(chunks)
    return KSelect(cfg, mgr, _cache=cache, _llm=llm)


# ── search() ─────────────────────────────────────────────────────────────────


def test_search_fast_returns_search_result(tmp_path):
    """search(fast=True) returns a SearchResult with hits."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(0)
    chunks = [_chunk(f"document {i}", rng.random(_DIM).tolist(), i) for i in range(20)]
    ks = _build_ks(chunks, tmp_path)

    fixed_emb = np.array(chunks[0].embedding, dtype="float32")
    ks._retrieval_engine.embed_query = MagicMock(return_value=fixed_emb)

    result = ks.search("document 0", fast=True)
    assert isinstance(result, SearchResult)
    assert len(result.hits) > 0
    for h in result.hits:
        assert isinstance(h.score, float)
        assert isinstance(h.snippet, str)


def test_search_multiple_flags_raises(tmp_path):
    """search() with two mode flags raises KSelectConfigError."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    from kselect.exceptions import ConfigError

    rng = np.random.default_rng(1)
    chunks = [_chunk(f"doc {i}", rng.random(_DIM).tolist(), i) for i in range(5)]
    ks = _build_ks(chunks, tmp_path)
    ks._retrieval_engine.embed_query = MagicMock(
        return_value=np.array(chunks[0].embedding, dtype="float32")
    )

    with pytest.raises(ConfigError):
        ks.search("query", fast=True, hybrid=True)


def test_search_diagnostics(tmp_path):
    """index_size(), index_drift(), recall_estimate() return sane values."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(2)
    chunks = [_chunk(f"doc {i}", rng.random(_DIM).tolist(), i) for i in range(50)]
    ks = _build_ks(chunks, tmp_path)

    assert ks.index_size() == 50
    assert ks.index_drift() == 0.0
    assert ks.recall_estimate() == 1.0


# ── query() ───────────────────────────────────────────────────────────────────


def test_query_returns_query_result(tmp_path):
    """query() with mock LLM returns a QueryResult with all fields."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(3)
    chunks = [_chunk(f"topic {i} explanation", rng.random(_DIM).tolist(), i) for i in range(20)]

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=("Mocked answer.", 0.95))

    ks = _build_ks(chunks, tmp_path, llm=mock_llm)
    fixed_emb = np.array(chunks[0].embedding, dtype="float32")
    ks._retrieval_engine.embed_query = MagicMock(return_value=fixed_emb)

    result = ks.query("what is topic 0?", fast=True)

    assert isinstance(result, QueryResult)
    assert result.answer == "Mocked answer."
    assert 0.0 <= result.confidence <= 1.0
    assert result.chunks_in_context <= result.chunks_retrieved
    assert result.context_tokens <= result.max_context_tokens
    assert result.chunks_dropped >= 0


def test_query_no_llm_raises(tmp_path):
    """query() without LLMClient raises LLMError."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    from kselect.exceptions import LLMError

    rng = np.random.default_rng(4)
    chunks = [_chunk(f"doc {i}", rng.random(_DIM).tolist(), i) for i in range(5)]
    ks = _build_ks(chunks, tmp_path)  # no LLM

    with pytest.raises(LLMError):
        ks.query("any question")


def test_query_context_tokens_within_budget(tmp_path):
    """context_tokens ≤ max_context_tokens in QueryResult."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    rng = np.random.default_rng(5)
    # Long chunks (~200 words each) to stress the token budget
    chunks = [
        _chunk(" ".join([f"word{j}"] * 200), rng.random(_DIM).tolist(), i)
        for i, j in enumerate(range(20))
    ]

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=("Answer.", 0.8))

    cfg = KSelectConfig()
    cfg.index.type = IndexType.FLAT
    cfg.context.max_context_tokens = 512

    backend = LocalBackend(str(tmp_path / "state"))
    mgr = IndexManager(FAISSIndex(), BM25Index(), backend, cfg)
    mgr.build(chunks)
    ks = KSelect(cfg, mgr, _llm=mock_llm)

    fixed_emb = np.array(chunks[0].embedding, dtype="float32")
    ks._retrieval_engine.embed_query = MagicMock(return_value=fixed_emb)

    result = ks.query("topic", fast=True)
    assert result.context_tokens <= cfg.context.max_context_tokens
