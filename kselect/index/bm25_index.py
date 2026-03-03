from __future__ import annotations

import json
import pickle
from pathlib import Path

from kselect.exceptions import IndexLoadError, IndexNotBuiltError, IndexSaveError
from kselect.models.chunk import Chunk
from kselect.models.config import BM25Config

_BM25_DIR = "bm25"


class BM25Index:
    """
    Wraps bm25s.BM25.
    Maintains a parallel id_map list aligned with the BM25 corpus.

    Serialization:
      {path}/bm25/model.pkl   → pickle of self._bm25
      {path}/bm25/id_map.json → json of self._id_map
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._id_map: list[str] = []
        self._corpus: list[str] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk], config: BM25Config) -> None:
        import bm25s  # type: ignore[import-untyped]

        self._bm25 = bm25s.BM25(k1=config.k1, b=config.b)
        self._corpus = [c.text for c in chunks]
        tokenized = bm25s.tokenize(self._corpus)
        self._bm25.index(tokenized)
        self._id_map = [c.id for c in chunks]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise IndexNotBuiltError("BM25Index.search() called before build() or load().")

        import bm25s  # type: ignore[import-untyped]

        k_clamped = min(k, len(self._id_map))
        if k_clamped == 0:
            return []

        tokenized_query = bm25s.tokenize([query])
        results, scores = self._bm25.retrieve(tokenized_query, k=k_clamped)

        # results shape: (1, k_clamped) — indices into corpus
        output: list[tuple[str, float]] = []
        for idx, score in zip(results[0], scores[0]):
            chunk_id = self._id_map[int(idx)]
            output.append((chunk_id, float(score)))

        return sorted(output, key=lambda x: x[1], reverse=True)

    # ── Incremental add ───────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> None:
        import bm25s  # type: ignore[import-untyped]

        new_texts = [c.text for c in chunks]
        self._corpus.extend(new_texts)
        self._id_map.extend(c.id for c in chunks)

        # Rebuild index over the full corpus
        if self._bm25 is None:
            self._bm25 = bm25s.BM25()
        tokenized = bm25s.tokenize(self._corpus)
        self._bm25.index(tokenized)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self._bm25 is None:
            raise IndexSaveError("Cannot save an unbuilt BM25Index.")
        d = Path(path) / _BM25_DIR
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.pkl").write_bytes(pickle.dumps(self._bm25))
            (d / "id_map.json").write_text(json.dumps(self._id_map))
            (d / "corpus.json").write_text(json.dumps(self._corpus))
        except Exception as exc:
            raise IndexSaveError(f"BM25Index.save() failed: {exc}") from exc

    def load(self, path: str) -> None:
        d = Path(path) / _BM25_DIR
        try:
            self._bm25 = pickle.loads((d / "model.pkl").read_bytes())  # noqa: S301
            self._id_map = json.loads((d / "id_map.json").read_text())
            corpus_file = d / "corpus.json"
            self._corpus = json.loads(corpus_file.read_text()) if corpus_file.exists() else []
        except Exception as exc:
            raise IndexLoadError(f"BM25Index.load() failed: {exc}") from exc

    @property
    def size(self) -> int:
        return len(self._id_map)
