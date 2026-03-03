from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path

import numpy as np

from kselect.backends.base import VectorBackend
from kselect.exceptions import IndexLoadError, IndexNotBuiltError, IndexSaveError
from kselect.index.bm25_index import BM25Index
from kselect.index.faiss_index import FAISSIndex
from kselect.models.chunk import Chunk
from kselect.models.config import KSelectConfig

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


class IndexManager:
    """
    Single owner of FAISSIndex + BM25Index.
    All code outside index/ interacts with IndexManager only.

    Thread safety: a RWLock pattern is approximated with a threading.Lock.
    Reads acquire the lock briefly; writes hold it for the full operation.
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index,
        backend: VectorBackend,
        config: KSelectConfig,
    ) -> None:
        self._faiss = faiss_index
        self._bm25 = bm25_index
        self._backend = backend
        self._config = config
        self._chunk_store: dict[str, Chunk] = {}
        self._lock = threading.Lock()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._faiss.build(chunks, self._config.index)
            if self._config.bm25.enabled:
                self._bm25.build(chunks, self._config.bm25)
            self._chunk_store = {c.id: c for c in chunks}

    # ── Search delegates ──────────────────────────────────────────────────────

    def search_faiss(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self._lock:
            return self._faiss.search(embedding, k)

    def search_bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._config.bm25.enabled:
            return []
        with self._lock:
            return self._bm25.search(query, k)

    # ── Chunk lookups ─────────────────────────────────────────────────────────

    def get_chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        with self._lock:
            return {
                cid: self._chunk_store[cid].text
                for cid in chunk_ids
                if cid in self._chunk_store
            }

    def get_chunk_embeddings(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        with self._lock:
            result: dict[str, np.ndarray] = {}
            for cid in chunk_ids:
                chunk = self._chunk_store.get(cid)
                if chunk and chunk.embedding is not None:
                    result[cid] = np.array(chunk.embedding, dtype="float32")
            return result

    def get_chunk_store(self) -> dict[str, Chunk]:
        with self._lock:
            return dict(self._chunk_store)

    # ── Incremental add ───────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        missing = [c.id for c in chunks if c.embedding is None]
        if missing:
            raise IndexNotBuiltError(
                f"add_chunks() called with {len(missing)} chunks missing embeddings."
            )

        with self._lock:
            self._faiss.add(chunks)
            if self._config.bm25.enabled:
                self._bm25.add(chunks)
            self._backend.upsert_chunks(chunks)
            for c in chunks:
                self._chunk_store[c.id] = c

        if self._faiss.needs_reindex:
            t = threading.Thread(target=self._background_reindex, daemon=True)
            t.start()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        tmp = path + "/.tmp_ks_save_" + uuid.uuid4().hex
        try:
            os.makedirs(tmp, exist_ok=True)
            self._faiss.save(tmp)
            self._bm25.save(tmp)

            # chunks.jsonl
            chunks_path = Path(tmp) / "chunks.jsonl"
            with chunks_path.open("w") as fh:
                for chunk in self._chunk_store.values():
                    fh.write(chunk.model_dump_json() + "\n")

            # config.json
            (Path(tmp) / "config.json").write_text(self._config.model_dump_json())

            # version.txt
            (Path(tmp) / "version.txt").write_text(_VERSION)

            # Atomic rename
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path)
            os.replace(tmp, path)

        except Exception as exc:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            raise IndexSaveError(f"IndexManager.save() failed: {exc}") from exc

    def load(self, path: str) -> None:
        base = Path(path)
        try:
            version = (base / "version.txt").read_text().strip()
            major = int(version.split(".")[0])
            if major != int(_VERSION.split(".")[0]):
                from kselect.exceptions import IndexLoadError
                raise IndexLoadError(
                    f"Version mismatch: index is v{version}, library is v{_VERSION}."
                )

            raw_config = json.loads((base / "config.json").read_text())
            self._config = KSelectConfig.model_validate(raw_config)

            self._faiss.load(path)
            if self._config.bm25.enabled:
                self._bm25.load(path)

            self._chunk_store = {}
            chunks_path = base / "chunks.jsonl"
            if chunks_path.exists():
                with chunks_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            c = Chunk.model_validate_json(line)
                            self._chunk_store[c.id] = c

        except (IndexLoadError, IndexSaveError):
            raise
        except Exception as exc:
            raise IndexLoadError(f"IndexManager.load() failed: {exc}") from exc

    # ── Background reindex ────────────────────────────────────────────────────

    def _background_reindex(self) -> None:
        logger.info("IndexManager: starting background FAISS reindex.")
        try:
            with self._lock:
                chunks = list(self._chunk_store.values())
                self._faiss.build(chunks, self._config.index)
            logger.info("IndexManager: background reindex complete (%d vectors).", len(chunks))
        except Exception as exc:
            logger.error("IndexManager: background reindex failed: %s", exc)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def index_drift(self) -> float:
        orig = self._faiss._original_size
        if orig == 0:
            return 0.0
        return (self._faiss.size - orig) / orig

    def recall_estimate(self) -> float:
        drift = self.index_drift()
        if drift <= 0.0:
            return 1.0
        _DRIFT_THRESHOLD = 0.20
        if drift <= _DRIFT_THRESHOLD:
            return 1.0 - 0.10 * (drift / 0.20)
        return max(0.0, 0.90 - 2.0 * (drift - 0.20))

    def index_size(self) -> int:
        return self._faiss.size
