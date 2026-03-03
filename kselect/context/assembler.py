from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tiktoken

from kselect.models.config import ContextConfig, ContextStrategy
from kselect.models.hit import Hit

if TYPE_CHECKING:
    from kselect.llm.base import LLMClient

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


class ContextAssembler:
    """
    Takes ranked hits and constructs the final context window for the LLM.
    Called only by query(), never by search().
    """

    def __init__(self, llm: "LLMClient | None" = None) -> None:
        self._llm = llm
        self._token_cache: dict[str, int] = {}

    def assemble(
        self,
        hits: list[Hit],
        config: ContextConfig,
    ) -> tuple[list[Hit], int]:
        strategy = str(config.strategy)
        max_tokens = config.max_context_tokens

        if strategy == ContextStrategy.LOST_IN_MIDDLE:
            return self._assemble_lost_in_middle(hits, max_tokens)
        elif strategy == ContextStrategy.SUMMARIZE_OVERFLOW:
            return self._assemble_summarize_overflow(hits, max_tokens)
        else:
            return self._assemble_score_order(hits, max_tokens)

    def _assemble_score_order(
        self, hits: list[Hit], max_tokens: int
    ) -> tuple[list[Hit], int]:
        selected: list[Hit] = []
        total = 0
        for hit in hits:
            tok = self._count_tokens_for_hit(hit)
            if total + tok > max_tokens:
                break
            selected.append(hit)
            total += tok
        return selected, total

    def _assemble_lost_in_middle(
        self, hits: list[Hit], max_tokens: int
    ) -> tuple[list[Hit], int]:
        if not hits:
            return [], 0
        odds = hits[::2]
        evens = hits[1::2]
        reordered = odds + list(reversed(evens))
        selected: list[Hit] = []
        total = 0
        for hit in reordered:
            tok = self._count_tokens_for_hit(hit)
            if total + tok > max_tokens:
                break
            selected.append(hit)
            total += tok
        return selected, total

    def _assemble_summarize_overflow(
        self, hits: list[Hit], max_tokens: int
    ) -> tuple[list[Hit], int]:
        if self._llm is None:
            logger.warning(
                "ContextAssembler: SUMMARIZE_OVERFLOW requires LLMClient; falling back to TRUNCATE."
            )
            return self._assemble_score_order(hits, max_tokens)

        budget_primary = int(max_tokens * 0.75)
        primary: list[Hit] = []
        total = 0
        overflow: list[Hit] = []
        for hit in hits:
            tok = self._count_tokens_for_hit(hit)
            if total + tok <= budget_primary:
                primary.append(hit)
                total += tok
            else:
                overflow.append(hit)

        if not overflow:
            return primary, total

        import asyncio
        overflow_text = "\n\n".join(h.snippet for h in overflow)
        summary_prompt = f"Summarize the following passages into a short paragraph:\n\n{overflow_text}"
        try:
            loop = asyncio.new_event_loop()
            summary_answer, _ = loop.run_until_complete(
                self._llm.generate(summary_prompt, [], max_tokens=256)
            )
            loop.close()
        except Exception as exc:
            logger.warning("ContextAssembler: summary LLM call failed (%s); truncating.", exc)
            return primary, total

        summary_hit = Hit(
            chunk_id="__summary__",
            doc_id="__summary__",
            score=0.0,
            snippet=summary_answer,
            metadata={},
            rank=len(primary) + 1,
        )
        summary_tok = self._count_tokens(summary_answer)
        remaining = max_tokens - total
        if summary_tok <= remaining:
            primary.append(summary_hit)
            total += summary_tok

        return primary, total

    def _count_tokens_for_hit(self, hit: Hit) -> int:
        if hit.chunk_id not in self._token_cache:
            self._token_cache[hit.chunk_id] = self._count_tokens(hit.snippet)
        return self._token_cache[hit.chunk_id]

    def _count_tokens(self, text: str) -> int:
        return len(_ENCODING.encode(text))
