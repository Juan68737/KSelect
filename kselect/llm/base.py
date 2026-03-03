from __future__ import annotations

from abc import ABC, abstractmethod

from kselect.models.hit import Hit


class LLMClient(ABC):

    @abstractmethod
    async def generate(
        self,
        query: str,
        context_chunks: list[Hit],
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """
        Generate an answer given a query and retrieved context.
        Returns (answer_text, confidence_score).
        Confidence derived from token logprobs when available;
        otherwise estimated from cross-encoder scores of context chunks.
        """
