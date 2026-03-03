from __future__ import annotations

import logging

from kselect.models.chunk import Chunk

logger = logging.getLogger(__name__)

_CANDIDATE_WARN_THRESHOLD = 200


class CrossEncoderReranker:
    """
    Wraps sentence_transformers.CrossEncoder.
    Model loaded once at __init__, reused across all queries.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            from kselect.exceptions import RankingError
            raise RankingError(
                "sentence-transformers is required for CrossEncoderReranker."
            ) from exc
        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        chunk_store: dict[str, Chunk],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        if len(candidates) > _CANDIDATE_WARN_THRESHOLD:
            logger.warning(
                "CrossEncoderReranker: %d candidates exceeds %d — may be slow.",
                len(candidates),
                _CANDIDATE_WARN_THRESHOLD,
            )

        pairs: list[tuple[str, str]] = []
        valid_ids: list[str] = []

        for chunk_id, _ in candidates:
            chunk = chunk_store.get(chunk_id)
            if chunk is None:
                logger.warning("CrossEncoderReranker: chunk_id %s not in chunk_store; skipping.", chunk_id)
                continue
            pairs.append((query, chunk.text))
            valid_ids.append(chunk_id)

        if not pairs:
            return []

        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(valid_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(cid, float(score)) for cid, score in ranked[:top_k]]
