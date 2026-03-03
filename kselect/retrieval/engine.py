from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from kselect.index.manager import IndexManager
from kselect.models.chunk import Chunk
from kselect.models.config import FusionMode, KSelectConfig
from kselect.models.trace import QueryTrace
from kselect.retrieval.fusion import rrf, weighted_fusion

logger = logging.getLogger(__name__)

_EMBED_MODEL_CACHE: dict[str, object] = {}


class RetrievalEngine:
    """
    Runs FAISS and BM25 searches in parallel, fuses results, returns candidates.
    Does NOT rank — that is RankingEngine's responsibility.
    """

    OVER_FETCH_FACTOR: int = 3

    def __init__(self, index_manager: IndexManager, config: KSelectConfig) -> None:
        self._manager = index_manager
        self._config = config

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int,
        fusion_mode: FusionMode,
        filters: dict | None = None,
        trace: QueryTrace | None = None,
    ) -> list[tuple[str, float]]:
        fetch_k = k * self.OVER_FETCH_FACTOR
        t0 = time.perf_counter()

        # ── Parallel search ───────────────────────────────────────────────────
        if str(fusion_mode) == str(FusionMode.DENSE):
            faiss_results = self._manager.search_faiss(query_embedding, fetch_k)
            bm25_results: list[tuple[str, float]] = []
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_future = pool.submit(self._manager.search_faiss, query_embedding, fetch_k)
                b_future = pool.submit(self._manager.search_bm25, query, fetch_k)
                faiss_results = f_future.result()
                bm25_results = b_future.result()

        # ── Metadata filters (post-retrieval) ─────────────────────────────────
        if filters:
            chunk_store = self._manager.get_chunk_store()
            faiss_results = _apply_filters(faiss_results, chunk_store, filters)
            bm25_results = _apply_filters(bm25_results, chunk_store, filters)

        # ── Fusion ────────────────────────────────────────────────────────────
        if str(fusion_mode) == str(FusionMode.RRF):
            results = rrf(
                faiss_results,
                bm25_results,
                k=self._config.fusion.rrf_k,
                top_n=fetch_k,
            )
        elif str(fusion_mode) == str(FusionMode.WEIGHTED):
            results = weighted_fusion(
                faiss_results,
                bm25_results,
                dense_weight=self._config.fusion.dense_weight,
                bm25_weight=self._config.fusion.bm25_weight,
                top_n=fetch_k,
            )
        else:
            results = faiss_results[:fetch_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if trace is not None:
            trace.faiss_candidates = len(faiss_results)
            trace.bm25_candidates = len(bm25_results)
            trace.after_fusion = len(results)
            trace.retrieval_latency_ms = elapsed_ms

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query using the same model as the index.
        Model is loaded once and cached at class level.
        """
        model_name = self._config.embedding.model
        if model_name not in _EMBED_MODEL_CACHE:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
                _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            except ImportError as exc:
                from kselect.exceptions import EmbeddingError
                raise EmbeddingError(
                    "sentence-transformers is required for query embedding."
                ) from exc

        import faiss  # type: ignore[import-untyped]

        model = _EMBED_MODEL_CACHE[model_name]
        embedding = model.encode(  # type: ignore[union-attr]
            [query],
            normalize_embeddings=self._config.embedding.normalize,
            show_progress_bar=False,
        )
        vec = np.array(embedding, dtype="float32")
        faiss.normalize_L2(vec)
        return vec[0]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _apply_filters(
    results: list[tuple[str, float]],
    chunk_store: dict[str, Chunk],
    filters: dict,
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for chunk_id, score in results:
        chunk = chunk_store.get(chunk_id)
        if chunk is None:
            continue
        meta = {**chunk.metadata.extra, "source_file": chunk.metadata.source_file}
        if all(meta.get(k) == v for k, v in filters.items()):
            out.append((chunk_id, score))
    return out
