from __future__ import annotations

import hashlib
import logging
from typing import Callable

from kselect.exceptions import IngestionError
from kselect.ingestion.chunking import Chunker
from kselect.ingestion.loaders import BaseLoader
from kselect.models.chunk import Chunk
from kselect.models.config import EmbeddingConfig, KSelectConfig

logger = logging.getLogger(__name__)

_LOG_INTERVAL = 1_000  # log progress every N chunks


class IngestionPipeline:
    """
    Orchestrates the full document → Chunk pipeline:
      BaseLoader.load() → Chunker.chunk() → Embedder.embed() → list[Chunk]

    Processing is batched: embed in groups of EmbeddingConfig.batch_size.
    Logs progress at INFO level every 1,000 chunks processed.

    remove_duplicates=True: compute SHA256(chunk.text) for each chunk.
    Chunks whose text hash matches a hash in the seen set (built from
    existing_chunk_ids or accumulated within this run) are skipped.
    """

    def run(
        self,
        loader: BaseLoader,
        config: KSelectConfig,
        existing_chunk_ids: set[str] | None = None,
    ) -> list[Chunk]:
        """
        existing_chunk_ids: set of text-hash IDs already in the index.
        Used for incremental runs to skip already-indexed content.
        Pass None for initial full builds.

        Returns list[Chunk] with embedding populated on every Chunk.
        Raises IngestionError if any document fails to parse,
        wrapping the original exception with the file path in the message.
        """
        # ── Load ──────────────────────────────────────────────────────────────
        try:
            docs = loader.load()
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Loader failed: {exc}") from exc

        # ── Chunk ─────────────────────────────────────────────────────────────
        embedder = _Embedder(config.embedding)
        chunker = Chunker(embed_fn=embedder.embed if config.chunking.strategy == "semantic" else None)

        all_chunks: list[Chunk] = []
        for text, meta in docs:
            source = meta.get("source_file", "<unknown>")
            try:
                chunks = chunker.chunk(text, meta, config.chunking)
            except Exception as exc:
                raise IngestionError(
                    f"Chunking failed for {source!r}: {exc}"
                ) from exc
            all_chunks.extend(chunks)

        # ── Dedup ─────────────────────────────────────────────────────────────
        if config.chunking.remove_duplicates:
            seen_hashes: set[str] = set(existing_chunk_ids or ())
            deduped: list[Chunk] = []
            for chunk in all_chunks:
                text_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    deduped.append(chunk)
                else:
                    logger.debug("Dedup: skipping duplicate chunk from %s", chunk.metadata.source_file)
            all_chunks = deduped

        # ── Embed ─────────────────────────────────────────────────────────────
        batch_size = config.embedding.batch_size
        total = len(all_chunks)

        for i in range(0, total, batch_size):
            batch = all_chunks[i : i + batch_size]

            # Skip embedding for chunks that already have a precomputed embedding
            # (from CSVLoader's vector_col feature)
            needs_embed = [c for c in batch if c.embedding is None]
            has_embed = [c for c in batch if c.embedding is not None]

            if needs_embed:
                texts = [c.text for c in needs_embed]
                embeddings = embedder.embed(texts)
                for chunk, emb in zip(needs_embed, embeddings):
                    chunk.embedding = emb

            done = min(i + batch_size, total)
            if done % _LOG_INTERVAL < batch_size or done == total:
                logger.info("IngestionPipeline: embedded %d / %d chunks", done, total)

        return all_chunks


# ── Internal embedder ─────────────────────────────────────────────────────────


class _Embedder:
    """
    Thin wrapper around sentence-transformers SentenceTransformer.
    Lazy-loaded on first call to embed().
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None  # lazy

    def _get_model(self):  # type: ignore[return]
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise IngestionError(
                "sentence-transformers is required for embedding. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._config.model)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        import numpy as np

        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()
