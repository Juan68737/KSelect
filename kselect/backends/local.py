from __future__ import annotations

from pathlib import Path

from kselect.backends.base import VectorBackend
from kselect.models.chunk import Chunk

_STORE_FILE = "chunks.jsonl"


class LocalBackend(VectorBackend):
    """
    Wraps a local directory containing serialized chunks.
    Used when KSelect is initialized via from_folder() / from_csv() / etc.
    The chunk store is a JSONL file: one Chunk JSON per line.
    """

    def __init__(self, state_dir: str) -> None:
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._store = self._dir / _STORE_FILE

    @classmethod
    def from_uri(cls, uri: str, **kwargs: object) -> "LocalBackend":
        """Parse "local://path/to/state_dir" → LocalBackend(path/to/state_dir)."""
        path = uri.removeprefix("local://")
        return cls(path)

    # ── VectorBackend interface ────────────────────────────────────────────────

    def get_all_chunks(self) -> list[Chunk]:
        if not self._store.exists():
            return []
        chunks: list[Chunk] = []
        with self._store.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate_json(line))
        return chunks

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        id_set = set(ids)
        return [c for c in self.get_all_chunks() if c.id in id_set]

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        with self._store.open("a") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")

    def delete_chunks(self, ids: list[str]) -> None:
        id_set = set(ids)
        remaining = [c for c in self.get_all_chunks() if c.id not in id_set]
        with self._store.open("w") as fh:
            for chunk in remaining:
                fh.write(chunk.model_dump_json() + "\n")

    def count(self) -> int:
        if not self._store.exists():
            return 0
        with self._store.open() as fh:
            return sum(1 for line in fh if line.strip())
