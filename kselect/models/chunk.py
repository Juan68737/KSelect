from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Arbitrary user-defined metadata attached to a chunk at ingestion time.
    All values must be JSON-serializable (str, int, float, bool, None).
    """

    model_config = {"extra": "allow"}

    source_file: str
    chunk_index: int  # position within parent document
    char_start: int
    char_end: int
    token_count: int
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """
    The atomic unit produced by IngestionPipeline and consumed by IndexManager.
    Each Chunk maps 1:1 to one FAISS vector and one BM25 document.
    """

    id: str  # uuid4, assigned at ingestion
    text: str  # raw text of this chunk
    embedding: list[float] | None = None  # populated after embed(); None before
    metadata: ChunkMetadata
