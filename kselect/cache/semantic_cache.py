from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from kselect.models.answer import QueryResult
from kselect.models.cache import CacheStats
from kselect.models.hit import SearchResult

if TYPE_CHECKING:
    from kselect.models.config import CacheConfig
    from kselect.ranking.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

_COST_PER_CALL_USD = 0.001


class SemanticCache:
    """
    Embedding-based semantic query cache backed by faiss.IndexFlatIP.
    TTL enforced lazily at read time. LRU eviction via deque.
    Thread-safe via threading.RLock.
    """

    class CacheEntry(BaseModel):
        query: str
        query_embedding: list[float]
        result: SearchResult | QueryResult
        created_at: float
        hit_count: int = 0

    def __init__(
        self,
        config: "CacheConfig",
        cross_encoder: "CrossEncoderReranker | None" = None,
    ) -> None:
        import faiss  # type: ignore[import-untyped]

        self._config = config
        self._cross_encoder = cross_encoder
        self._lock = threading.RLock()

        # FAISS index is built lazily on first set()
        self._faiss_index: object | None = None
        self._dim: int | None = None

        # positional index → CacheEntry
        self._entries: dict[int, SemanticCache.CacheEntry] = {}
        # LRU order: front = oldest
        self._lru: deque[int] = deque()
        # positions that have been logically evicted
        self._evicted: set[int] = set()
        self._next_pos: int = 0

        # Stats counters
        self._hits: int = 0
        self._misses: int = 0
        self._false_positives: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        query_embedding: np.ndarray,
        query: str,
        config: "CacheConfig",
    ) -> SearchResult | QueryResult | None:
        with self._lock:
            if self._faiss_index is None or self._next_pos == 0:
                self._misses += 1
                return None

            vec = self._to_row(query_embedding)
            distances, indices = self._faiss_index.search(vec, 1)  # type: ignore[union-attr]
            similarity = float(distances[0][0])
            pos = int(indices[0][0])

            if pos < 0 or pos in self._evicted:
                self._misses += 1
                return None

            entry = self._entries.get(pos)
            if entry is None:
                self._misses += 1
                return None

            now = time.time()

            if similarity >= config.similarity_threshold:
                # TTL check
                if (now - entry.created_at) > config.ttl_seconds:
                    self._evict(pos)
                    self._misses += 1
                    return None
                entry.hit_count += 1
                self._hits += 1
                return entry.result

            # Borderline verification
            if (
                config.verify_borderline
                and config.borderline_lower <= similarity < config.similarity_threshold
            ):
                if self._cross_encoder is None:
                    self._misses += 1
                    return None
                try:
                    score = self._cross_encoder._model.predict([(query, entry.query)])[0]
                    if score > 0.85:
                        entry.hit_count += 1
                        self._hits += 1
                        return entry.result
                    else:
                        self._false_positives += 1
                except Exception as exc:
                    logger.warning("SemanticCache: borderline cross-encoder failed: %s", exc)

            self._misses += 1
            return None

    def set(
        self,
        query_embedding: np.ndarray,
        query: str,
        result: SearchResult | QueryResult,
    ) -> None:
        with self._lock:
            import faiss  # type: ignore[import-untyped]

            vec = self._to_row(query_embedding)

            # Initialise index on first set
            if self._faiss_index is None:
                self._dim = vec.shape[1]
                self._faiss_index = faiss.IndexFlatIP(self._dim)

            # Evict LRU if at capacity
            if len(self._entries) - len(self._evicted) >= self._config.max_size:
                while self._lru:
                    oldest = self._lru.popleft()
                    if oldest not in self._evicted:
                        self._evict(oldest)
                        break

            pos = self._next_pos
            self._next_pos += 1

            self._faiss_index.add(vec)  # type: ignore[union-attr]
            entry = SemanticCache.CacheEntry(
                query=query,
                query_embedding=query_embedding.tolist(),
                result=result,
                created_at=time.time(),
            )
            self._entries[pos] = entry
            self._lru.append(pos)

            # Rebuild index if >50% evicted
            if len(self._evicted) > self._next_pos * 0.5 and self._next_pos > 10:
                self._rebuild_index(faiss)

    def stats(self) -> CacheStats:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            fp_denom = self._hits + self._false_positives
            fp_rate = self._false_positives / fp_denom if fp_denom > 0 else 0.0
            return CacheStats(
                hit_rate=hit_rate,
                false_positive_rate=fp_rate,
                size=len(self._entries) - len(self._evicted),
                llm_calls_saved=self._hits,
                cost_saved_usd=self._hits * _COST_PER_CALL_USD,
                qps_saved=0.0,
            )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _evict(self, pos: int) -> None:
        self._evicted.add(pos)
        self._entries.pop(pos, None)

    def _to_row(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.array(embedding, dtype="float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return vec

    def _rebuild_index(self, faiss_mod: object) -> None:
        """Rebuild FAISS index from active (non-evicted) entries."""
        active_positions = [p for p in self._entries if p not in self._evicted]
        if not active_positions:
            self._faiss_index = faiss_mod.IndexFlatIP(self._dim)  # type: ignore[union-attr]
            self._evicted.clear()
            return

        new_index = faiss_mod.IndexFlatIP(self._dim)  # type: ignore[union-attr]
        for new_pos, old_pos in enumerate(active_positions):
            entry = self._entries[old_pos]
            vec = np.array(entry.query_embedding, dtype="float32").reshape(1, -1)
            new_index.add(vec)
            self._entries[new_pos] = entry

        # Remove old entries that were remapped
        for old_pos in active_positions:
            if old_pos >= len(active_positions):
                del self._entries[old_pos]

        self._faiss_index = new_index
        self._evicted.clear()
        self._next_pos = len(active_positions)
        self._lru = deque(range(self._next_pos))
