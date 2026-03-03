from __future__ import annotations


def rrf(
    faiss_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke, Buettcher — SIGIR 2009).

    score(d) = Σ 1 / (k + rank(d, list_i))   (rank is 1-indexed)
    Documents missing from a list contribute 0 from that list.

    Returns: (chunk_id, rrf_score) sorted descending, no duplicates.
    """
    scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(faiss_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        result = result[:top_n]

    return result


def weighted_fusion(
    faiss_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """
    Weighted linear combination after min-max score normalization.

    normalized_score = (score - min) / (max - min)
    If max == min: all normalized = 1.0.

    Raises ValueError if weights don't sum to 1.0 (tolerance 1e-6).
    Returns: (chunk_id, combined_score) sorted descending.
    """
    if abs(dense_weight + bm25_weight - 1.0) > 1e-6:
        raise ValueError(
            f"dense_weight + bm25_weight must equal 1.0, got {dense_weight + bm25_weight}"
        )

    def _normalize(results: list[tuple[str, float]]) -> dict[str, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        denom = hi - lo if hi != lo else 1.0
        return {cid: (s - lo) / denom for cid, s in results}

    dense_norm = _normalize(faiss_results)
    bm25_norm = _normalize(bm25_results)

    all_ids = set(dense_norm) | set(bm25_norm)
    combined: dict[str, float] = {
        cid: dense_weight * dense_norm.get(cid, 0.0) + bm25_weight * bm25_norm.get(cid, 0.0)
        for cid in all_ids
    }

    result = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        result = result[:top_n]

    return result
