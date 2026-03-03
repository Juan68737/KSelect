from __future__ import annotations

import os
from urllib.parse import urlparse

from kselect.backends.base import VectorBackend
from kselect.exceptions import BackendConnectionError, BackendReadError, BackendWriteError
from kselect.models.chunk import Chunk, ChunkMetadata

_FETCH_BATCH = 1_000   # Pinecone list()+fetch() limit
_UPSERT_BATCH = 100    # Pinecone upsert limit


class PineconeBackend(VectorBackend):
    """
    Wraps pinecone-client v4+.
    get_all_chunks: uses list() + fetch() in batches of 1,000 (Pinecone API limit).
    upsert_chunks: uses upsert() in batches of 100 vectors.
    Metadata stored as Pinecone metadata dict alongside the vector.
    """

    def __init__(self, index_name: str, namespace: str = "") -> None:
        self._index_name = index_name
        self._namespace = namespace
        self._index = None  # lazy-initialized

    @classmethod
    def from_uri(cls, uri: str, **kwargs: object) -> "PineconeBackend":
        """
        Parse "pinecone://index-name/namespace"
        API key read from PINECONE_API_KEY env var.
        """
        parsed = urlparse(uri)
        index_name = parsed.netloc
        namespace = parsed.path.lstrip("/")
        return cls(index_name=index_name, namespace=namespace)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_index(self):  # type: ignore[return]
        if self._index is not None:
            return self._index
        try:
            from pinecone import Pinecone  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendConnectionError(
                "pinecone-client is required for PineconeBackend. "
                "Install it with: pip install 'kselect[pinecone]'"
            ) from exc
        api_key = os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise BackendConnectionError(
                "PINECONE_API_KEY environment variable is not set."
            )
        try:
            pc = Pinecone(api_key=api_key)
            self._index = pc.Index(self._index_name)
        except Exception as exc:
            raise BackendConnectionError(
                f"Cannot connect to Pinecone index '{self._index_name}': {exc}"
            ) from exc
        return self._index

    @staticmethod
    def _chunk_to_vector(chunk: Chunk) -> dict:
        meta: dict = {
            "text": chunk.text,
            "source_file": chunk.metadata.source_file,
            "chunk_index": chunk.metadata.chunk_index,
            "char_start": chunk.metadata.char_start,
            "char_end": chunk.metadata.char_end,
            "token_count": chunk.metadata.token_count,
            **chunk.metadata.extra,
        }
        return {
            "id": chunk.id,
            "values": chunk.embedding or [],
            "metadata": meta,
        }

    @staticmethod
    def _vector_to_chunk(vec: dict) -> Chunk:
        meta = vec.get("metadata", {})
        return Chunk(
            id=vec["id"],
            text=meta.get("text", ""),
            embedding=vec.get("values"),
            metadata=ChunkMetadata(
                source_file=meta.get("source_file", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", 0)),
                token_count=int(meta.get("token_count", 0)),
                extra={
                    k: v for k, v in meta.items()
                    if k not in {"text", "source_file", "chunk_index",
                                 "char_start", "char_end", "token_count"}
                },
            ),
        )

    # ── VectorBackend interface ────────────────────────────────────────────────

    def get_all_chunks(self) -> list[Chunk]:
        idx = self._get_index()
        chunks: list[Chunk] = []
        try:
            # list() returns an iterable of ID batches
            for id_batch in idx.list(namespace=self._namespace, limit=_FETCH_BATCH):
                ids = list(id_batch) if not isinstance(id_batch, list) else id_batch
                if not ids:
                    continue
                result = idx.fetch(ids=ids, namespace=self._namespace)
                for vec in result.vectors.values():
                    chunks.append(self._vector_to_chunk(vec))
        except Exception as exc:
            raise BackendReadError(f"get_all_chunks (Pinecone) failed: {exc}") from exc
        return chunks

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        idx = self._get_index()
        chunks: list[Chunk] = []
        try:
            for start in range(0, len(ids), _FETCH_BATCH):
                batch_ids = ids[start : start + _FETCH_BATCH]
                result = idx.fetch(ids=batch_ids, namespace=self._namespace)
                for vec in result.vectors.values():
                    chunks.append(self._vector_to_chunk(vec))
        except Exception as exc:
            raise BackendReadError(f"get_chunks_by_ids (Pinecone) failed: {exc}") from exc
        return chunks

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        idx = self._get_index()
        try:
            for start in range(0, len(chunks), _UPSERT_BATCH):
                batch = chunks[start : start + _UPSERT_BATCH]
                vectors = [self._chunk_to_vector(c) for c in batch]
                idx.upsert(vectors=vectors, namespace=self._namespace)
        except Exception as exc:
            raise BackendWriteError(f"upsert_chunks (Pinecone) failed: {exc}") from exc

    def delete_chunks(self, ids: list[str]) -> None:
        if not ids:
            return
        idx = self._get_index()
        try:
            for start in range(0, len(ids), _FETCH_BATCH):
                batch_ids = ids[start : start + _FETCH_BATCH]
                idx.delete(ids=batch_ids, namespace=self._namespace)
        except Exception as exc:
            raise BackendWriteError(f"delete_chunks (Pinecone) failed: {exc}") from exc

    def count(self) -> int:
        idx = self._get_index()
        try:
            stats = idx.describe_index_stats()
            ns = stats.namespaces.get(self._namespace)
            return int(ns.vector_count) if ns else 0
        except Exception as exc:
            raise BackendReadError(f"count (Pinecone) failed: {exc}") from exc
