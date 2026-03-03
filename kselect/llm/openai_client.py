from __future__ import annotations

import logging

from kselect.exceptions import LLMConnectionError, LLMError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from kselect.llm.base import LLMClient
from kselect.models.hit import Hit

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    "Be concise and accurate. If the context does not contain enough information, say so."
)


class OpenAIClient(LLMClient):
    """
    LLMClient implementation backed by the OpenAI chat completions API.
    Compatible with any OpenAI-compatible endpoint (set base_url).
    Requires: pip install openai
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LLMError(
                "openai package is required: pip install openai"
            ) from exc

        self._model = model
        self._temperature = temperature
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def generate(
        self,
        query: str,
        context_chunks: list[Hit],
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """
        Returns (answer_text, confidence).
        Confidence derived from logprobs when available; else defaults to 0.0.
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
            from openai import APIConnectionError, APITimeoutError, RateLimitError  # type: ignore[import-untyped]

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=self._temperature,
                logprobs=True,
                top_logprobs=1,
            )

        except RateLimitError as exc:
            raise LLMRateLimitError(f"OpenAI rate limit: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI timeout: {exc}") from exc
        except APIConnectionError as exc:
            raise LLMConnectionError(f"OpenAI connection error: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        answer = choice.message.content or ""

        # Derive confidence from mean logprob
        confidence = 0.0
        try:
            if choice.logprobs and choice.logprobs.content:
                import math
                lps = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
                if lps:
                    mean_lp = sum(lps) / len(lps)
                    confidence = float(math.exp(mean_lp))
        except Exception:
            pass

        if not answer:
            raise LLMResponseError("OpenAI returned an empty response.")

        return answer, confidence
