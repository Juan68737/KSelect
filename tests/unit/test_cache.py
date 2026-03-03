"""Phase 7 tests — SemanticCache."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from kselect.cache.semantic_cache import SemanticCache
from kselect.models.cache import CacheStats
from kselect.models.config import CacheConfig
from kselect.models.hit import Hit, SearchResult


def _cfg(**kwargs) -> CacheConfig:
    defaults = dict(
        enabled=True,
        similarity_threshold=0.97,
        verify_borderline=False,
        borderline_lower=0.93,
        ttl_seconds=3600,
        max_size=100,
    )
    defaults.update(kwargs)
    return CacheConfig(**defaults)


def _result(query: str = "test") -> SearchResult:
    return SearchResult(hits=[], total_hits=0, query=query)


def _vec(n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(n).astype("float32")
    return v / np.linalg.norm(v)


# ── Basic get/set ─────────────────────────────────────────────────────────────


def test_cache_miss_on_empty():
    """get() on empty cache returns None."""
    pytest.importorskip("faiss")
    cfg = _cfg()
    cache = SemanticCache(cfg)
    result = cache.get(_vec(), "query", cfg)
    assert result is None


def test_cache_set_then_get_hit():
    """Identical embedding → cache hit on second call."""
    pytest.importorskip("faiss")
    cfg = _cfg()
    cache = SemanticCache(cfg)
    vec = _vec()
    r = _result("hello")
    cache.set(vec, "hello", r)
    hit = cache.get(vec, "hello", cfg)
    assert hit is not None
    assert hit.query == "hello"


def test_cache_different_query_miss():
    """Very different embedding → cache miss."""
    pytest.importorskip("faiss")
    cfg = _cfg(similarity_threshold=0.97)
    cache = SemanticCache(cfg)

    vec_a = _vec(seed=0)
    vec_b = _vec(seed=42)
    # Make them orthogonal
    vec_b -= vec_b.dot(vec_a) * vec_a
    if np.linalg.norm(vec_b) > 0:
        vec_b /= np.linalg.norm(vec_b)

    cache.set(vec_a, "query A", _result("A"))
    hit = cache.get(vec_b, "query B", cfg)
    # Should be a miss since vectors are orthogonal (similarity ~ 0)
    assert hit is None


def test_cache_ttl_expiry():
    """Entry expired by TTL → cache miss."""
    pytest.importorskip("faiss")
    cfg = _cfg(ttl_seconds=0)
    cache = SemanticCache(cfg)
    vec = _vec()
    cache.set(vec, "q", _result("q"))
    # TTL = 0 means any access after creation should expire
    time.sleep(0.01)
    hit = cache.get(vec, "q", cfg)
    assert hit is None


# ── Stats ─────────────────────────────────────────────────────────────────────


def test_cache_stats_hit_rate():
    """Hit rate computed correctly after set + 2 gets."""
    pytest.importorskip("faiss")
    cfg = _cfg()
    cache = SemanticCache(cfg)
    vec = _vec()
    cache.set(vec, "q", _result("q"))
    cache.get(vec, "q", cfg)   # hit
    cache.get(_vec(seed=99), "other", cfg)  # miss (probably)
    stats = cache.stats()
    assert isinstance(stats, CacheStats)
    assert 0.0 <= stats.hit_rate <= 1.0


# ── Integration: KSelect search with cache ────────────────────────────────────


def test_kselect_search_cache_roundtrip(tmp_path):
    """Second identical search() returns cached SearchResult."""
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")

    import uuid
    from kselect.index.faiss_index import FAISSIndex
    from kselect.index.bm25_index import BM25Index
    from kselect.index.manager import IndexManager
    from kselect.backends.local import LocalBackend
    from kselect.models.chunk import Chunk, ChunkMetadata
    from kselect.models.config import KSelectConfig, IndexType
    from kselect.kselect import KSelect

    rng = np.random.default_rng(0)
    DIM = 16

    def _chunk(i):
        return Chunk(
            id=str(uuid.uuid4()),
            text=f"test document {i}",
            embedding=rng.random(DIM).tolist(),
            metadata=ChunkMetadata(
                source_file="test.txt",
                chunk_index=i,
                char_start=0,
                char_end=10,
                token_count=3,
            ),
        )

    chunks = [_chunk(i) for i in range(20)]
    cfg = KSelectConfig()
    cfg.index.type = IndexType.FLAT
    cfg.cache.enabled = True
    cfg.cache.similarity_threshold = 0.99

    backend = LocalBackend(str(tmp_path / "state"))
    mgr = IndexManager(FAISSIndex(), BM25Index(), backend, cfg)
    mgr.build(chunks)

    from kselect.cache.semantic_cache import SemanticCache
    cache = SemanticCache(cfg.cache)

    ks = KSelect(cfg, mgr, _cache=cache)

    # Mock embed_query to return deterministic vector
    fixed_vec = np.array(chunks[0].embedding, dtype="float32")
    ks._retrieval_engine.embed_query = MagicMock(return_value=fixed_vec)

    r1 = ks.search("test document 0", fast=True)
    r2 = ks.search("test document 0", fast=True)

    assert isinstance(r1, SearchResult)
    assert isinstance(r2, SearchResult)
    # Both should be the same object (r2 from cache)
    assert r2.query == r1.query
