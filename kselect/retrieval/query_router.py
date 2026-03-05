"""
Query Router — detects aggregation vs lookup intent and returns optimal retrieval params.

Aggregation queries ("list all", "highest/lowest", "ages of", "how many") need:
  - High k (fetch many candidates — you need multiple rows to aggregate)
  - WEIGHTED fusion (BM25-heavy: exact field matches dominate)
  - fast=True ranking (no MMR — you WANT near-duplicate rows because they're all answers)
  - Larger context window (pack in as many matching rows as possible)

Lookup queries ("who is", "what was the fare of X", "did Y survive") need:
  - Normal k
  - RRF fusion (balanced FAISS + BM25)
  - hybrid=True (cross-encoder + MMR — find the one right row, no duplicates)
  - Normal context window
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_AGGREGATION_PATTERNS = [
    # List/enumerate patterns
    r"\blist\b", r"\blist all\b", r"\benumerate\b", r"\bshow all\b", r"\bfind all\b",
    r"\ball (the )?(passengers?|rows?|entries|records|people)\b",
    # Superlative/extreme value patterns
    r"\bhighest\b", r"\blowest\b", r"\bmost\b", r"\bleast\b", r"\bbiggest\b",
    r"\bsmallest\b", r"\blargest\b", r"\bmaximum\b", r"\bminimum\b",
    r"\btop \d+\b", r"\bbottom \d+\b",
    # Aggregation/count patterns
    r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\baverage\b", r"\bmean\b",
    r"\bsum\b", r"\bnumber of\b",
    # Multi-value retrieval patterns
    r"\bages? of\b", r"\bnames? of\b", r"\bfares? of\b",
    r"\bwho (were|are|did|didn'?t|survived?|died?)\b",
    r"\bwhich (passengers?|people|rows?)\b",
    r"\bwhat were\b", r"\bwhat are\b",
]

_AGGREGATION_RE = re.compile(
    "|".join(_AGGREGATION_PATTERNS),
    re.IGNORECASE,
)


@dataclass
class RoutingParams:
    k: int
    hybrid: bool
    fast: bool
    fusion: str         # "rrf" | "weighted"
    max_context_tokens: int
    query_type: str     # "aggregation" | "lookup"


def route_query(question: str, base_k: int = 15) -> RoutingParams:
    """
    Classify question and return optimal retrieval parameters.

    Aggregation: needs many rows packed into context.
    Lookup: needs the single best matching row, diverse context.
    """
    is_aggregation = bool(_AGGREGATION_RE.search(question))

    if is_aggregation:
        return RoutingParams(
            # Fetch 4x more candidates — for "list all survivors in 1st class"
            # you need as many matching rows as possible, not just top-15.
            k=min(base_k * 4, 60),
            hybrid=False,
            fast=True,          # No MMR — for aggregation you WANT all matching rows
            fusion="weighted",  # BM25-heavy: exact field matches ("Survived: 1", "Pclass: 1") dominate
            max_context_tokens=10000,  # Pack in as many rows as the budget allows
            query_type="aggregation",
        )
    else:
        return RoutingParams(
            k=base_k,
            hybrid=True,        # Cross-encoder + MMR: find the one best row, no duplicates
            fast=False,
            fusion="rrf",       # Balanced FAISS + BM25
            max_context_tokens=8000,
            query_type="lookup",
        )
