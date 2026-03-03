from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mmr_diversify(
    candidates: list[tuple[str, float]],
    chunk_embeddings: dict[str, np.ndarray],
    top_k: int,
    lambda_: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    MMR(d) = lambda_ * relevance(d) - (1 - lambda_) * max_sim(d, selected)

    relevance(d) = rerank score normalized to [0, 1].
    max_sim = max cosine similarity between d and any already-selected chunk.

    lambda_=1.0 → pure relevance (no-op).
    lambda_=0.0 → pure diversity.

    Complexity: O(top_k * n). Returns top_k results in selection order.
    Missing embeddings treated as max_sim=0 (maximally diverse). Logs WARNING.
    """
    if not candidates:
        return []

    # Normalize relevance scores to [0, 1]
    raw_scores = [s for _, s in candidates]
    lo, hi = min(raw_scores), max(raw_scores)
    denom = hi - lo if hi != lo else 1.0
    relevance: dict[str, float] = {
        cid: (s - lo) / denom for cid, s in candidates
    }

    remaining = [cid for cid, _ in candidates]
    selected: list[tuple[str, float]] = []
    selected_embeddings: list[np.ndarray] = []

    for _ in range(min(top_k, len(candidates))):
        best_id: str | None = None
        best_score = float("-inf")

        for cid in remaining:
            rel = relevance[cid]

            if not selected_embeddings:
                max_sim = 0.0
            elif cid not in chunk_embeddings:
                logger.warning("mmr_diversify: no embedding for chunk %s; treating max_sim=0", cid)
                max_sim = 0.0
            else:
                emb = chunk_embeddings[cid]
                sims = [float(np.dot(emb, sel)) for sel in selected_embeddings]
                max_sim = max(sims)

            score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_id = cid

        if best_id is None:
            break

        selected.append((best_id, best_score))
        remaining.remove(best_id)

        if best_id in chunk_embeddings:
            selected_embeddings.append(chunk_embeddings[best_id])

    return selected
