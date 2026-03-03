from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse, parse_qs

from kselect.backends.base import VectorBackend
from kselect.exceptions import BackendConnectionError, BackendReadError, BackendWriteError
from kselect.models.chunk import Chunk, ChunkMetadata

_BATCH_SIZE = 10_000


class PGVectorBackend(VectorBackend):
    """
    Wraps a PostgreSQL table with a pgvector `vector(N)` column.

    Uses psycopg3 (the `psycopg` package, NOT psycopg2).
    Sync connection pool for indexing; async pool for serving.

    KSelect NEVER writes DDL. If the schema doesn't match, raises
    BackendConnectionError with a descriptive message.
    """

    def __init__(
        self,
        dsn: str,
        table: str,
        text_col: str = "content",
        metadata_cols: list[str] | None = None,
    ) -> None:
        self._dsn = dsn
        self._table = table
        self._text_col = text_col
        self._metadata_cols: list[str] = metadata_cols or []
        self._conn: Any = None  # lazy-connected

    @classmethod
    def from_uri(
        cls,
        uri: str,
        text_col: str = "content",
        metadata_cols: list[str] | None = None,
    ) -> "PGVectorBackend":
        """
        Parse "pgvector://host/dbname?table=legal_cases"
        Reconstructs DSN as: postgresql://host/dbname
        Table name extracted from query string.
        """
        parsed = urlparse(uri)
        qs = parse_qs(parsed.query)
        table = qs.get("table", [parsed.path.lstrip("/")])[0]
        dsn = f"postgresql://{parsed.netloc}{parsed.path}"
        return cls(dsn=dsn, table=table, text_col=text_col, metadata_cols=metadata_cols)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _connect(self) -> Any:
        try:
            import psycopg  # type: ignore[import-untyped]
        except ImportError as exc:
            raise BackendConnectionError(
                "psycopg is required for PGVectorBackend. "
                "Install it with: pip install 'kselect[pgvector]'"
            ) from exc
        try:
            conn = psycopg.connect(self._dsn)
        except Exception as exc:
            raise BackendConnectionError(f"Cannot connect to PostgreSQL: {exc}") from exc
        return conn

    @property
    def _db(self) -> Any:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _row_to_chunk(self, row: Any, cur: Any) -> Chunk:
        col_names = [desc.name for desc in cur.description]
        row_dict = dict(zip(col_names, row))
        chunk_id = str(row_dict["chunk_id"])
        text = row_dict[self._text_col]
        embedding_raw = row_dict.get("embedding")
        embedding = list(map(float, embedding_raw)) if embedding_raw is not None else None
        meta_extra: dict[str, Any] = {}
        for col in self._metadata_cols:
            if col in row_dict:
                meta_extra[col] = row_dict[col]
        metadata = ChunkMetadata(
            source_file=str(row_dict.get("source_file", "")),
            chunk_index=int(row_dict.get("chunk_index", 0)),
            char_start=int(row_dict.get("char_start", 0)),
            char_end=int(row_dict.get("char_end", len(text))),
            token_count=int(row_dict.get("token_count", 0)),
            extra=meta_extra,
        )
        return Chunk(id=chunk_id, text=text, embedding=embedding, metadata=metadata)

    # ── VectorBackend interface ────────────────────────────────────────────────

    def get_all_chunks(self) -> list[Chunk]:
        try:
            with self._db.cursor(name="kselect_full_scan") as cur:
                cur.execute(f"SELECT * FROM {self._table}")  # noqa: S608
                chunks: list[Chunk] = []
                while True:
                    rows = cur.fetchmany(_BATCH_SIZE)
                    if not rows:
                        break
                    for row in rows:
                        chunks.append(self._row_to_chunk(row, cur))
                return chunks
        except Exception as exc:
            raise BackendReadError(f"get_all_chunks failed: {exc}") from exc

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM {self._table} WHERE chunk_id = ANY(%s)",  # noqa: S608
                    (ids,),
                )
                return [self._row_to_chunk(row, cur) for row in cur.fetchall()]
        except Exception as exc:
            raise BackendReadError(f"get_chunks_by_ids failed: {exc}") from exc

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        try:
            with self._db.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        f"""
                        INSERT INTO {self._table}
                            (chunk_id, {self._text_col}, embedding,
                             source_file, chunk_index, char_start, char_end, token_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE
                            SET {self._text_col} = EXCLUDED.{self._text_col},
                                embedding        = EXCLUDED.embedding
                        """,  # noqa: S608
                        (
                            chunk.id,
                            chunk.text,
                            chunk.embedding,
                            chunk.metadata.source_file,
                            chunk.metadata.chunk_index,
                            chunk.metadata.char_start,
                            chunk.metadata.char_end,
                            chunk.metadata.token_count,
                        ),
                    )
            self._db.commit()
        except Exception as exc:
            self._db.rollback()
            raise BackendWriteError(f"upsert_chunks failed: {exc}") from exc

    def delete_chunks(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE chunk_id = ANY(%s)",  # noqa: S608
                    (ids,),
                )
            self._db.commit()
        except Exception as exc:
            self._db.rollback()
            raise BackendWriteError(f"delete_chunks failed: {exc}") from exc

    def count(self) -> int:
        try:
            with self._db.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table}")  # noqa: S608
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            raise BackendReadError(f"count failed: {exc}") from exc
