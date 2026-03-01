from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    One cited source in a QueryResult.
    """

    chunk_id: str
    doc_id: str
    snippet: str
    score: float
    metadata: dict


class QueryResult(BaseModel):
    """
    Return type of KSelect.query(). Extends SearchResult with LLM output.
    """

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source]
    query: str

    # Context window diagnostics
    chunks_retrieved: int
    chunks_in_context: int
    chunks_dropped: int
    context_tokens: int
    max_context_tokens: int

    trace: "QueryTrace | None" = None


# Avoid circular import — QueryTrace is resolved at runtime via forward ref
from kselect.models.trace import QueryTrace  # noqa: E402, F401

QueryResult.model_rebuild()
