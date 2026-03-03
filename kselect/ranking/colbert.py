from __future__ import annotations

import logging

import numpy as np

from kselect.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """
    Late interaction reranker using token-level embeddings (MaxSim).
    Uses colbert-ai library (kselect[rerank]).

    If colbert-ai is not installed: falls back to CrossEncoderReranker automatically.

    MaxSim score(q, d) = Σ_i max_j sim(q_i, d_j)
    where q_i = query token embeddings, d_j = document token embeddings.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0") -> None:
        self._model_name = model_name
        self._model = None
        self._fallback = None

        try:
            from colbert import Searcher  # type: ignore[import-untyped]  # noqa: F401
            from colbert.modeling.checkpoint import Checkpoint  # type: ignore[import-untyped]
            self._model = Checkpoint(model_name, colbert_config=None)
        except ImportError:
            logger.warning(
                "colbert-ai not installed; ColBERTReranker will fall back to "
                "CrossEncoderReranker. Install with: pip install 'kselect[rerank]'"
            )
            from kselect.ranking.cross_encoder import CrossEncoderReranker
            self._fallback = CrossEncoderReranker()

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        chunk_store: dict[str, Chunk],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if self._fallback is not None:
            return self._fallback.rerank(query, candidates, chunk_store, top_k)

        return self._colbert_rerank(query, candidates, chunk_store, top_k)

    def _colbert_rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        chunk_store: dict[str, Chunk],
        top_k: int,
    ) -> list[tuple[str, float]]:
        import torch  # type: ignore[import-untyped]

        model = self._model
        q_encodings = model.queryFromText([query])  # type: ignore[union-attr]

        scored: list[tuple[str, float]] = []
        for chunk_id, _ in candidates:
            chunk = chunk_store.get(chunk_id)
            if chunk is None:
                logger.warning("ColBERTReranker: chunk_id %s not in chunk_store; skipping.", chunk_id)
                continue
            d_encodings = model.docFromText([chunk.text])  # type: ignore[union-attr]

            # MaxSim: sum over query tokens of max similarity with doc tokens
            q_emb = q_encodings[0]  # (q_len, dim)
            d_emb = d_encodings[0]  # (d_len, dim)
            sims = torch.matmul(q_emb, d_emb.T)  # (q_len, d_len)
            score = float(sims.max(dim=1).values.sum())
            scored.append((chunk_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
