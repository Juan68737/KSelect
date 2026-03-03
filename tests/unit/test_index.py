"""Phase 3 tests — FAISSIndex and BM25Index."""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from kselect.index.faiss_index import FAISSIndex
from kselect.index.bm25_index import BM25Index
from kselect.models.chunk import Chunk, ChunkMetadata
from kselect.models.config import IndexConfig, IndexType, BM25Config


def _make_chunk(text: str, embedding: list[float]) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            source_file="test.txt",
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            token_count=len(text.split()),
        ),
    )


def _random_chunks(n: int, dim: int = 16) -> list[Chunk]:
    rng = np.random.default_rng(42)
    return [
        _make_chunk(f"document about topic {i}", rng.random(dim).tolist())
        for i in range(n)
    ]


# ── FAISSIndex ────────────────────────────────────────────────────────────────


def test_faiss_build_search_roundtrip():
    """Build index from 100 chunks, search returns expected chunk_id at rank 1."""
    pytest.importorskip("faiss")
    dim = 16
    chunks = _random_chunks(100, dim)

    # Make one chunk's embedding clearly point in a specific direction
    target = chunks[0]
    target.embedding = [1.0] + [0.0] * (dim - 1)

    config = IndexConfig(type=IndexType.FLAT)
    idx = FAISSIndex()
    idx.build(chunks, config)

    # Query with the same direction — should return target at rank 1
    query = np.array([1.0] + [0.0] * (dim - 1), dtype="float32")
    results = idx.search(query, k=5)

    assert results, "Expected non-empty search results"
    assert results[0][0] == target.id, f"Expected target at rank 1, got {results[0][0]}"


def test_faiss_save_load(tmp_path):
    """Save and reload FAISSIndex; search results identical before and after."""
    pytest.importorskip("faiss")
    dim = 16
    chunks = _random_chunks(50, dim)
    config = IndexConfig(type=IndexType.FLAT)

    idx = FAISSIndex()
    idx.build(chunks, config)

    query = np.array(chunks[0].embedding, dtype="float32")
    results_before = idx.search(query, k=3)

    idx.save(str(tmp_path))

    idx2 = FAISSIndex()
    idx2.load(str(tmp_path))
    results_after = idx2.search(query, k=3)

    assert [cid for cid, _ in results_before] == [cid for cid, _ in results_after]


def test_faiss_drift_threshold():
    """add() sets needs_reindex=True after >20% of original size is added."""
    pytest.importorskip("faiss")
    dim = 16
    n = 100
    chunks = _random_chunks(n, dim)
    config = IndexConfig(type=IndexType.FLAT)

    idx = FAISSIndex()
    idx.build(chunks, config)
    assert not idx.needs_reindex

    # Add 21 chunks (21% drift)
    extra = _random_chunks(21, dim)
    idx.add(extra)
    assert idx.needs_reindex, "needs_reindex should be True after >20% drift"


# ── BM25Index ─────────────────────────────────────────────────────────────────


def test_bm25_exact_term_retrieval():
    """BM25 ranks chunk with exact query term at position 0."""
    pytest.importorskip("bm25s")
    texts = [
        "The cat sat on the mat",
        "Dogs are great companions",
        "Python is a programming language",
        "Aurora borealis lights the night sky",
    ]
    chunks = [_make_chunk(t, [0.0] * 4) for t in texts]

    config = BM25Config(k1=1.5, b=0.75)
    idx = BM25Index()
    idx.build(chunks, config)

    results = idx.search("Python programming", k=4)
    assert results, "Expected non-empty BM25 results"
    top_id = results[0][0]
    top_chunk = next(c for c in chunks if c.id == top_id)
    assert "Python" in top_chunk.text or "programming" in top_chunk.text
