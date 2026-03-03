from __future__ import annotations

from urllib.parse import urlparse

from kselect.backends.base import VectorBackend
from kselect.exceptions import BackendConnectionError, BackendReadError, BackendWriteError
from kselect.models.chunk import Chunk, ChunkMetadata


class ChromaDBBackend(VectorBackend):
    """
    Wraps chromadb v0.5+.
    Client mode: http (if uri contains a host) or embedded (local path).
    Collection name extracted from uri path component.
    """

    def __init__(self, collection_name: str, host: str = "", port: int = 8000) -> None:
        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._collection = None  # lazy-initialized

    @classmethod
    def from_uri(cls, uri: str, **kwargs: object) -> "ChromaDBBackend":
        """
        "chromadb://collection_name"           → embedded, collection=collection_name
        "chromadb://host:8000/collection_name" → http client
        """
        parsed = urlparse(uri)
        netloc = parsed.netloc  # e.g. "" or "host:8000"
        path = parsed.path.lstrip("/")

        if not netloc or ":" not in netloc and "." not in netloc and netloc == netloc:
            # No real host — treat netloc as collection name (no-host form)
            if path:
                # chromadb://host:8000/collection → http
                host_part = netloc
                host, _, port_str = host_part.partition(":")
                port = int(port_str) if port_str else 8000
                return cls(collection_name=path, host=host, port=port)
            else:
                # chromadb://collection_name (no slash)
                return cls(collection_name=netloc, host="", port=8000)
        else:
            host_part = netloc
            host, _, port_str = host_part.partition(":")
            port = int(port_str) if port_str else 8000
            return cls(collection_name=path, host=host, port=port)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_collection(self):  # type: ignore[return]
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendConnectionError(
                "chromadb is required for ChromaDBBackend. "
                "Install it with: pip install 'kselect[chromadb]'"
            ) from exc
        try:
            if self._host:
                client = chromadb.HttpClient(host=self._host, port=self._port)
            else:
                client = chromadb.EphemeralClient()
            self._collection = client.get_or_create_collection(self._collection_name)
        except Exception as exc:
            raise BackendConnectionError(
                f"Cannot connect to ChromaDB collection '{self._collection_name}': {exc}"
            ) from exc
        return self._collection

    @staticmethod
    def _doc_to_chunk(doc_id: str, document: str, embedding, metadata: dict) -> Chunk:
        return Chunk(
            id=doc_id,
            text=document,
            embedding=list(map(float, embedding)) if embedding is not None else None,
            metadata=ChunkMetadata(
                source_file=metadata.get("source_file", ""),
                chunk_index=int(metadata.get("chunk_index", 0)),
                char_start=int(metadata.get("char_start", 0)),
                char_end=int(metadata.get("char_end", len(document))),
                token_count=int(metadata.get("token_count", 0)),
                extra={
                    k: v for k, v in metadata.items()
                    if k not in {"source_file", "chunk_index", "char_start",
                                 "char_end", "token_count"}
                },
            ),
        )

    # ── VectorBackend interface ────────────────────────────────────────────────

    def get_all_chunks(self) -> list[Chunk]:
        col = self._get_collection()
        try:
            result = col.get(include=["documents", "embeddings", "metadatas"])
            chunks: list[Chunk] = []
            for doc_id, doc, emb, meta in zip(
                result["ids"],
                result["documents"],
                result["embeddings"],
                result["metadatas"],
            ):
                chunks.append(self._doc_to_chunk(doc_id, doc, emb, meta or {}))
            return chunks
        except Exception as exc:
            raise BackendReadError(f"get_all_chunks (ChromaDB) failed: {exc}") from exc

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        col = self._get_collection()
        try:
            result = col.get(
                ids=ids,
                include=["documents", "embeddings", "metadatas"],
            )
            chunks: list[Chunk] = []
            for doc_id, doc, emb, meta in zip(
                result["ids"],
                result["documents"],
                result["embeddings"],
                result["metadatas"],
            ):
                chunks.append(self._doc_to_chunk(doc_id, doc, emb, meta or {}))
            return chunks
        except Exception as exc:
            raise BackendReadError(f"get_chunks_by_ids (ChromaDB) failed: {exc}") from exc

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        col = self._get_collection()
        try:
            col.upsert(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=[c.embedding for c in chunks],  # type: ignore[arg-type]
                metadatas=[
                    {
                        "source_file": c.metadata.source_file,
                        "chunk_index": c.metadata.chunk_index,
                        "char_start": c.metadata.char_start,
                        "char_end": c.metadata.char_end,
                        "token_count": c.metadata.token_count,
                        **c.metadata.extra,
                    }
                    for c in chunks
                ],
            )
        except Exception as exc:
            raise BackendWriteError(f"upsert_chunks (ChromaDB) failed: {exc}") from exc

    def delete_chunks(self, ids: list[str]) -> None:
        if not ids:
            return
        col = self._get_collection()
        try:
            col.delete(ids=ids)
        except Exception as exc:
            raise BackendWriteError(f"delete_chunks (ChromaDB) failed: {exc}") from exc

    def count(self) -> int:
        col = self._get_collection()
        try:
            return col.count()
        except Exception as exc:
            raise BackendReadError(f"count (ChromaDB) failed: {exc}") from exc
