from __future__ import annotations

from abc import ABC, abstractmethod

from kselect.models.chunk import Chunk


class VectorBackend(ABC):
    """
    Abstract interface over any vector store.
    All backends must implement these five methods.
    IndexManager calls these; it does not call backend-specific methods directly.
    """

    @abstractmethod
    def get_all_chunks(self) -> list[Chunk]:
        """
        Return all chunks with embeddings populated.
        Called during initial index build from an existing backend.
        For large backends, implementations MUST stream in batches of 10,000.
        Never load the full table into memory.
        """

    @abstractmethod
    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Return specific chunks by chunk_id. Used during reranking for text lookup."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """
        Write chunks to the backend. Called by add_doc() / add_folder().
        For LocalBackend: no-op (FAISS is the source of truth).
        For remote backends: write vectors + metadata to the database.
        """

    @abstractmethod
    def delete_chunks(self, ids: list[str]) -> None:
        """Delete chunks by chunk_id. Used for document removal."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks stored."""
