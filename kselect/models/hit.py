from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Hit(BaseModel):
    """
    One ranked result returned by search().
    """

    chunk_id: str
    doc_id: str  # source_file from ChunkMetadata
    score: float  # final score after all ranking stages
    snippet: str  # chunk.text (or truncated)
    metadata: dict[str, Any]  # passthrough of ChunkMetadata.extra
    rank: int  # 1-indexed position in result list

    # Diagnostic fields — populated when trace=True
    faiss_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class SearchResult(BaseModel):
    """
    Return type of KSelect.search().
    """

    hits: list[Hit]
    total_hits: int
    query: str
    trace: "QueryTrace | None" = None


# Avoid circular import — QueryTrace is resolved at runtime via forward ref
from kselect.models.trace import QueryTrace  # noqa: E402, F401

SearchResult.model_rebuild()
