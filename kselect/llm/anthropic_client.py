from __future__ import annotations

import logging
import math

from kselect.exceptions import LLMConnectionError, LLMError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from kselect.llm.base import LLMClient
from kselect.models.hit import Hit

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise data analyst. Answer the question using ONLY facts and values "
    "explicitly stated in the provided context passages. "
    "Cite specific names, numbers, and field values from the context directly in your answer. "
    "Do not infer, generalize, or add information not present in the context. "
    "If the context does not contain enough information to answer, say exactly: "
    "'The provided context does not contain enough information to answer this question.'"
)


class AnthropicClient(LLMClient):
    """
    LLMClient implementation backed by the Anthropic Messages API.
    Requires: pip install anthropic
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LLMError(
                "anthropic package is required: pip install anthropic"
            ) from exc

        self._model = model
        self._temperature = temperature
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        query: str,
        context_chunks: list[Hit],
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """
        Returns (answer_text, confidence).
        Anthropic does not expose token logprobs, so confidence is estimated
        from the mean retrieval score of the context chunks (0–1 range).
        """
        context_text = "\n\n".join(
            f"[{i+1}] {h.snippet}" for i, h in enumerate(context_chunks)
        )
        user_message = (
            f"Context:\n{context_text}\n\nQuestion: {query}"
            if context_text
            else query
        )

        try:
            import anthropic  # type: ignore[import-untyped]

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=self._temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(f"Anthropic rate limit: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic timeout: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(f"Anthropic connection error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        answer = response.content[0].text if response.content else ""
        if not answer:
            raise LLMResponseError("Anthropic returned an empty response.")

        # Estimate confidence from retrieval scores.
        # Raw cosine scores from a flat IP index cluster in a low absolute range
        # (e.g. 0.01–0.10 for short texts), so we rank-normalize them to [0, 1]
        # using the top score as the ceiling so the best chunk reads as 1.0 and
        # weaker matches scale down proportionally.
        confidence = 0.0
        if context_chunks:
            scores = [h.score for h in context_chunks if h.score is not None]
            if scores:
                top = max(scores)
                if top > 0:
                    mean_normalized = sum(s / top for s in scores) / len(scores)
                    confidence = float(round(mean_normalized, 2))

        return answer, confidence
