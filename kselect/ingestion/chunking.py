from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable

from kselect.exceptions import ChunkingError
from kselect.models.chunk import Chunk, ChunkMetadata
from kselect.models.config import ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class Chunker:
    """
    Stateless transformer: (text, metadata_dict) → list[Chunk].
    Strategy selected via ChunkingConfig.strategy.

    All strategies enforce min_chunk_length: chunks shorter than this threshold
    are merged with the previous chunk rather than discarded or kept as-is.

    Each Chunk receives a uuid4 id assigned here. chunk_index is the 0-based
    position within the parent document.

    embed_fn: optional callable (texts: list[str]) → list[list[float]].
              Required only when strategy=SEMANTIC.
    """

    def __init__(self, embed_fn: Callable[[list[str]], list[list[float]]] | None = None) -> None:
        self._embed_fn = embed_fn

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any],
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Dispatch to the appropriate private strategy method."""
        strategy = config.strategy

        try:
            if strategy == ChunkingStrategy.SLIDING_WINDOW:
                raw_chunks = self._sliding_window(text, config.chunk_size, config.chunk_overlap)
            elif strategy == ChunkingStrategy.SENTENCE:
                raw_chunks = self._sentence(text, config.chunk_size, config.chunk_overlap)
            elif strategy == ChunkingStrategy.SEMANTIC:
                raw_chunks = self._semantic(text, config.semantic_threshold)
            elif strategy == ChunkingStrategy.PARAGRAPH:
                raw_chunks = self._paragraph(text)
            else:
                raise ChunkingError(f"Unknown chunking strategy: {strategy!r}")
        except ChunkingError:
            raise
        except Exception as exc:
            raise ChunkingError(
                f"Chunking strategy {strategy!r} failed: {exc}"
            ) from exc

        raw_chunks = self._enforce_min_length(raw_chunks, config.min_chunk_length)

        source_file = metadata.get("source_file", "")
        extra = {k: v for k, v in metadata.items() if k != "source_file"}

        chunks: list[Chunk] = []
        char_pos = 0
        for idx, chunk_text in enumerate(raw_chunks):
            char_start = text.find(chunk_text, char_pos)
            if char_start == -1:
                char_start = char_pos
            char_end = char_start + len(chunk_text)
            char_pos = char_start

            token_count = _count_tokens(chunk_text)

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    embedding=None,
                    metadata=ChunkMetadata(
                        source_file=source_file,
                        chunk_index=idx,
                        char_start=char_start,
                        char_end=char_end,
                        token_count=token_count,
                        extra=extra,
                    ),
                )
            )

        return chunks

    # ── Strategies ─────────────────────────────────────────────────────────────

    def _sliding_window(self, text: str, size: int, overlap: int) -> list[str]:
        """
        Token-based sliding window using tiktoken with cl100k_base encoding.
        Encodes full text → splits token list → decodes each window.
        overlap is in tokens, not characters.
        """
        enc = _get_tiktoken_encoder()
        tokens = enc.encode(text)

        if not tokens:
            return []

        stride = max(1, size - overlap)
        windows: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + size, len(tokens))
            window_tokens = tokens[start:end]
            windows.append(enc.decode(window_tokens))
            if end >= len(tokens):
                break
            start += stride

        return windows

    def _sentence(self, text: str, size: int, overlap_sentences: int) -> list[str]:
        """
        NLTK punkt_tab sentence tokenizer.
        Packs sentences greedily until the token budget (size) is full.
        overlap_sentences: number of sentences from the end of the previous
        chunk to prepend to the next chunk.
        """
        try:
            import nltk  # type: ignore[import-untyped]
            try:
                sentences: list[str] = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(text)
        except ImportError as exc:
            raise ChunkingError(
                "nltk is required for sentence chunking. Install it with: pip install nltk"
            ) from exc

        if not sentences:
            return []

        enc = _get_tiktoken_encoder()

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(enc.encode(sent))
            if current and current_tokens + sent_tokens > size:
                chunks.append(" ".join(current))
                # overlap: carry over the last N sentences
                if overlap_sentences > 0:
                    carry = current[-overlap_sentences:]
                    current = carry[:]
                    current_tokens = sum(len(enc.encode(s)) for s in current)
                else:
                    current = []
                    current_tokens = 0
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _semantic(self, text: str, threshold: float) -> list[str]:
        """
        Embedding-based semantic chunking.
        1. Tokenize into sentences with NLTK.
        2. Embed each sentence with the configured embedding model.
        3. Compute cosine similarity between adjacent sentence pairs.
        4. Insert a chunk boundary where similarity < threshold (topic shift).
        5. Merge resulting chunks smaller than min_chunk_length.

        WARNING: O(n_sentences) embedding calls. Expensive without GPU.
        Logs a warning if called on text with >500 sentences.
        """
        if self._embed_fn is None:
            raise ChunkingError(
                "embed_fn is required for semantic chunking. "
                "Pass an embedding callable to Chunker(embed_fn=...)."
            )

        try:
            import nltk  # type: ignore[import-untyped]
            try:
                sentences: list[str] = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(text)
        except ImportError as exc:
            raise ChunkingError(
                "nltk is required for semantic chunking. Install it with: pip install nltk"
            ) from exc

        if len(sentences) > 500:
            logger.warning(
                "Chunker._semantic: text has %d sentences — this may be slow without GPU.",
                len(sentences),
            )

        if not sentences:
            return []

        try:
            import numpy as np
            embeddings = np.array(self._embed_fn(sentences))

            # Cosine similarity between adjacent sentences
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            normed = embeddings / norms

            boundaries: list[int] = [0]
            for i in range(len(normed) - 1):
                sim = float(np.dot(normed[i], normed[i + 1]))
                if sim < threshold:
                    boundaries.append(i + 1)
            boundaries.append(len(sentences))

        except ImportError as exc:
            raise ChunkingError(
                "numpy is required for semantic chunking. Install it with: pip install numpy"
            ) from exc

        chunks: list[str] = []
        for start, end in zip(boundaries, boundaries[1:]):
            chunks.append(" ".join(sentences[start:end]))

        return [c for c in chunks if c.strip()]

    def _paragraph(self, text: str) -> list[str]:
        """
        Split on double newline (\\n\\n).
        Normalize whitespace within each paragraph.
        Short paragraphs are merged with the previous one.
        """
        raw = re.split(r"\n\n+", text)
        paragraphs = [" ".join(p.split()) for p in raw if p.strip()]
        return paragraphs  # min_length merge handled by _enforce_min_length

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _enforce_min_length(chunks: list[str], min_length: int) -> list[str]:
        """
        Merge chunks shorter than min_length with the previous chunk.
        If the first chunk is short, it stays (no previous to merge into).
        """
        if not chunks or min_length <= 0:
            return chunks

        result: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk) < min_length:
                result[-1] = result[-1] + " " + chunk
            else:
                result.append(chunk)
        return result


# ── Module-level helpers ───────────────────────────────────────────────────────

_TIKTOKEN_ENCODER = None


def _get_tiktoken_encoder():  # type: ignore[return]
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is not None:
        return _TIKTOKEN_ENCODER
    try:
        import tiktoken  # type: ignore[import-untyped]
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        return _TIKTOKEN_ENCODER
    except ImportError as exc:
        raise ChunkingError(
            "tiktoken is required for sliding-window and sentence chunking. "
            "Install it with: pip install tiktoken"
        ) from exc


def _count_tokens(text: str) -> int:
    try:
        enc = _get_tiktoken_encoder()
        return len(enc.encode(text))
    except ChunkingError:
        # Fallback: rough estimate
        return len(text.split())
