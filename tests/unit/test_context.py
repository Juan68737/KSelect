"""Phase 6 tests — ContextAssembler."""
from __future__ import annotations

import pytest

from kselect.context.assembler import ContextAssembler
from kselect.models.config import ContextConfig, ContextStrategy
from kselect.models.hit import Hit


def _hit(chunk_id: str, snippet: str, score: float = 1.0, rank: int = 1) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id="test.txt",
        score=score,
        snippet=snippet,
        metadata={},
        rank=rank,
    )


def _hits(n: int, words_each: int = 10) -> list[Hit]:
    return [
        _hit(f"c{i}", " ".join([f"word{i}"] * words_each), score=1.0 / (i + 1), rank=i + 1)
        for i in range(n)
    ]


# ── SCORE_ORDER ───────────────────────────────────────────────────────────────


def test_score_order_fits_all():
    """All hits fit within token budget → all returned."""
    assembler = ContextAssembler()
    hits = _hits(3, words_each=5)
    config = ContextConfig(strategy=ContextStrategy.SCORE_ORDER, max_context_tokens=10_000)
    selected, total = assembler.assemble(hits, config)
    assert len(selected) == 3
    assert total > 0


def test_score_order_truncates_on_budget():
    """Hits that exceed token budget are dropped."""
    assembler = ContextAssembler()
    # Each hit has ~100 words → ~100 tokens; budget = 50 → only first hit fits
    hits = _hits(5, words_each=100)
    config = ContextConfig(strategy=ContextStrategy.SCORE_ORDER, max_context_tokens=50)
    selected, total = assembler.assemble(hits, config)
    assert len(selected) < 5
    assert total <= 50


# ── LOST_IN_MIDDLE ────────────────────────────────────────────────────────────


def test_lost_in_middle_reorders():
    """LOST_IN_MIDDLE places rank-1 hit at position 0."""
    assembler = ContextAssembler()
    hits = _hits(6, words_each=5)
    config = ContextConfig(strategy=ContextStrategy.LOST_IN_MIDDLE, max_context_tokens=10_000)
    selected, _ = assembler.assemble(hits, config)
    # Rank-1 hit (c0) should be at position 0
    assert selected[0].chunk_id == "c0"


def test_lost_in_middle_empty():
    """LOST_IN_MIDDLE with no hits returns empty list."""
    assembler = ContextAssembler()
    config = ContextConfig(strategy=ContextStrategy.LOST_IN_MIDDLE, max_context_tokens=1000)
    selected, total = assembler.assemble([], config)
    assert selected == []
    assert total == 0


# ── TRUNCATE ──────────────────────────────────────────────────────────────────


def test_truncate_same_as_score_order():
    """TRUNCATE produces same result as SCORE_ORDER."""
    assembler = ContextAssembler()
    hits = _hits(5, words_each=50)
    budget = 200
    cfg_score = ContextConfig(strategy=ContextStrategy.SCORE_ORDER, max_context_tokens=budget)
    cfg_trunc = ContextConfig(strategy=ContextStrategy.TRUNCATE, max_context_tokens=budget)
    sel_score, tok_score = assembler.assemble(hits, cfg_score)
    sel_trunc, tok_trunc = assembler.assemble(hits, cfg_trunc)
    assert [h.chunk_id for h in sel_score] == [h.chunk_id for h in sel_trunc]
    assert tok_score == tok_trunc


# ── Token counting ────────────────────────────────────────────────────────────


def test_count_tokens_cached():
    """Token cache avoids double-encoding the same chunk_id."""
    assembler = ContextAssembler()
    hit = _hit("x", "hello world")
    config = ContextConfig(strategy=ContextStrategy.SCORE_ORDER, max_context_tokens=10_000)
    assembler.assemble([hit], config)
    assembler.assemble([hit], config)  # second call: should hit cache
    assert "x" in assembler._token_cache
