from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from kselect.exceptions import IndexLoadError, IndexNotBuiltError, IndexSaveError
from kselect.models.chunk import Chunk
from kselect.models.config import IndexConfig, IndexType

logger = logging.getLogger(__name__)

_DRIFT_THRESHOLD = 0.20
_FAISS_DIR = "faiss"


class FAISSIndex:
    """
    Wraps a faiss.Index. Owns the positional id→chunk_id mapping.

    Internal state:
      _index:              faiss.Index
      _id_map:             list[str]        position → chunk_id
      _chunk_id_to_pos:    dict[str, int]   chunk_id → position
      _original_size:      int              size at last full build
      _needs_reindex:      bool
    """

    def __init__(self) -> None:
        self._index = None
        self._id_map: list[str] = []
        self._chunk_id_to_pos: dict[str, int] = {}
        self._original_size: int = 0
        self._needs_reindex: bool = False
        self._index_type: str = IndexType.VLQ_ADC

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk], config: IndexConfig) -> None:
        import faiss  # type: ignore[import-untyped]

        missing = [c.id for c in chunks if c.embedding is None]
        if missing:
            raise IndexNotBuiltError(
                f"build() called with {len(missing)} chunks that have embedding=None."
            )

        dim = len(chunks[0].embedding)  # type: ignore[arg-type]
        self._index_type = str(config.type)
        self._index = self._make_index(config, dim, faiss)

        embeddings = np.array([c.embedding for c in chunks], dtype="float32")
        faiss.normalize_L2(embeddings)

        if str(config.type) in {
            str(IndexType.IVF_PQ128),
            str(IndexType.VLQ_ADC),
            str(IndexType.FCVI),
        }:
            self._index.train(embeddings)

        self._index.add(embeddings)
        self._id_map = [c.id for c in chunks]
        self._chunk_id_to_pos = {cid: i for i, cid in enumerate(self._id_map)}
        self._original_size = len(chunks)
        self._needs_reindex = False

    @staticmethod
    def _make_index(config: IndexConfig, dim: int, faiss):
        t = str(config.type)
        if t == str(IndexType.FLAT):
            return faiss.IndexFlatIP(dim)
        if t == str(IndexType.IVF_PQ128):
            quantizer = faiss.IndexFlatL2(dim)
            return faiss.IndexIVFPQ(quantizer, dim, config.nlist, config.m, 8)
        if t == str(IndexType.HNSW_SQ):
            inner = faiss.IndexHNSWFlat(dim, config.hnsw_m)
            return faiss.IndexIDMap2(inner)
        # VLQ_ADC and FCVI: treat as IVF_PQ128 for Phase 3
        quantizer = faiss.IndexFlatL2(dim)
        return faiss.IndexIVFPQ(quantizer, dim, config.nlist, config.m, 8)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._index is None:
            raise IndexNotBuiltError("search() called before build() or load().")

        import faiss  # type: ignore[import-untyped]

        q = query_embedding.copy().reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)

        k_clamped = min(k, self._index.ntotal)
        if k_clamped == 0:
            return []

        scores, indices = self._index.search(q, k_clamped)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._id_map[idx], float(score)))

        return results

    # ── Incremental add ───────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> None:
        if self._index is None:
            raise IndexNotBuiltError("add() called before build() or load().")

        import faiss  # type: ignore[import-untyped]

        embeddings = np.array([c.embedding for c in chunks], dtype="float32")
        faiss.normalize_L2(embeddings)

        if str(self._index_type) in {
            str(IndexType.IVF_PQ128),
            str(IndexType.VLQ_ADC),
            str(IndexType.FCVI),
        }:
            logger.warning(
                "FAISSIndex.add(): IVF index does not support true incremental add; "
                "recall may degrade above 20%% drift."
            )

        start_pos = len(self._id_map)
        self._index.add(embeddings)
        for i, chunk in enumerate(chunks):
            pos = start_pos + i
            self._id_map.append(chunk.id)
            self._chunk_id_to_pos[chunk.id] = pos

        if self._original_size > 0:
            drift = (self.size - self._original_size) / self._original_size
            if drift > _DRIFT_THRESHOLD:
                self._needs_reindex = True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import faiss  # type: ignore[import-untyped]

        if self._index is None:
            raise IndexSaveError("Cannot save an unbuilt FAISSIndex.")

        d = Path(path) / _FAISS_DIR
        try:
            d.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(d / "index.faiss"))
            (d / "id_map.json").write_text(json.dumps(self._id_map))
            (d / "meta.json").write_text(
                json.dumps({
                    "original_size": self._original_size,
                    "index_type": self._index_type,
                })
            )
        except Exception as exc:
            raise IndexSaveError(f"FAISSIndex.save() failed: {exc}") from exc

    def load(self, path: str) -> None:
        import faiss  # type: ignore[import-untyped]

        d = Path(path) / _FAISS_DIR
        try:
            self._index = faiss.read_index(str(d / "index.faiss"))
            self._id_map = json.loads((d / "id_map.json").read_text())
            meta = json.loads((d / "meta.json").read_text())
            self._original_size = meta["original_size"]
            self._index_type = meta["index_type"]
            self._chunk_id_to_pos = {cid: i for i, cid in enumerate(self._id_map)}
            self._needs_reindex = False
        except Exception as exc:
            raise IndexLoadError(f"FAISSIndex.load() failed: {exc}") from exc

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def needs_reindex(self) -> bool:
        return self._needs_reindex
