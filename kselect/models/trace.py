from __future__ import annotations

from pydantic import BaseModel


class QueryTrace(BaseModel):
    """
    Per-query timing and diagnostic data. Only populated when trace=True.
    """

    cache_hit: bool = False
    cache_similarity: float | None = None

    retrieval_latency_ms: float = 0.0  # FAISS + BM25 parallel search
    fusion_latency_ms: float = 0.0  # RRF or weighted fusion
    rerank_latency_ms: float = 0.0  # cross-encoder / ColBERT
    mmr_latency_ms: float = 0.0
    context_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    faiss_candidates: int = 0
    bm25_candidates: int = 0
    after_fusion: int = 0
    after_rerank: int = 0
    after_mmr: int = 0
    chunks_in_context: int = 0

    embedding_model: str = ""
    index_type: str = ""
    ranking_mode: str = ""
    fusion_mode: str = ""
