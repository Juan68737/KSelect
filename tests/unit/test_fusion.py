"""Phase 3 tests — fusion algorithms and MMR."""
from __future__ import annotations

import numpy as np
import pytest

from kselect.retrieval.fusion import rrf, weighted_fusion
from kselect.ranking.mmr import mmr_diversify


# ── rrf ───────────────────────────────────────────────────────────────────────


def test_rrf_no_duplicates():
    """RRF output contains no duplicate chunk_ids."""
    faiss = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("b", 5.0), ("c", 4.0), ("d", 3.0)]
    result = rrf(faiss, bm25)
    ids = [cid for cid, _ in result]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in RRF output"


def test_rrf_sorted_desc():
    """RRF output is sorted by rrf_score descending."""
    faiss = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("a", 5.0), ("b", 4.0), ("c", 3.0)]
    result = rrf(faiss, bm25)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_doc_in_one_list_only():
    """Doc appearing in only one list still appears in output with correct score."""
    faiss = [("a", 0.9), ("b", 0.8)]
    bm25 = [("c", 5.0), ("a", 4.0)]   # "b" missing from bm25, "c" missing from faiss
    result = rrf(faiss, bm25)
    ids = {cid for cid, _ in result}
    assert "b" in ids, "'b' (only in FAISS) should appear in RRF output"
    assert "c" in ids, "'c' (only in BM25) should appear in RRF output"

    # Score for "b" = 1/(60+2) only; score for "a" = 1/(60+1) + 1/(60+2)
    score_a = next(s for cid, s in result if cid == "a")
    score_b = next(s for cid, s in result if cid == "b")
    assert score_a > score_b, "Doc in both lists should outscore doc in one list"


def test_weighted_fusion_sum_to_one():
    """weighted_fusion raises ValueError when weights don't sum to 1.0."""
    faiss = [("a", 0.9)]
    bm25 = [("a", 5.0)]
    with pytest.raises(ValueError, match="1.0"):
        weighted_fusion(faiss, bm25, dense_weight=0.5, bm25_weight=0.6)


# ── MMR ───────────────────────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def test_mmr_lambda_one():
    """MMR with lambda_=1.0 returns same order as input (pure relevance)."""
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    embs = {
        "a": _unit(np.array([1.0, 0.0])),
        "b": _unit(np.array([0.0, 1.0])),
        "c": _unit(np.array([1.0, 1.0])),
    }
    result = mmr_diversify(candidates, embs, top_k=3, lambda_=1.0)
    # With lambda_=1.0 MMR score = relevance only → same order
    assert [cid for cid, _ in result] == ["a", "b", "c"]


def test_mmr_lambda_zero():
    """MMR with lambda_=0.0 selects maximally diverse chunks."""
    # Two near-identical vectors and one orthogonal — with lambda_=0 the
    # second selection should be the orthogonal one, not the near-duplicate.
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    embs = {
        "a": _unit(np.array([1.0, 0.0])),
        "b": _unit(np.array([1.0, 0.01])),   # nearly identical to "a"
        "c": _unit(np.array([0.0, 1.0])),     # orthogonal to "a"
    }
    result = mmr_diversify(candidates, embs, top_k=3, lambda_=0.0)
    ids = [cid for cid, _ in result]
    # After selecting "a" (tied highest relevance=1 when all normalized equally),
    # lambda_=0 → pick most diverse from selected → "c" before "b"
    assert ids.index("c") < ids.index("b"), (
        "With lambda_=0, orthogonal chunk 'c' should be picked before near-duplicate 'b'"
    )
