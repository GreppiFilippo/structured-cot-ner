from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 60.0


class LLMClient(ABC):
    """Backend-agnostic interface for a single-prompt text completion call."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        thinking_budget: int,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        raise NotImplementedError


class AsyncOpenAICompatibleClient(LLMClient):
    """Works with any OpenAI-compatible endpoint — here, a local vLLM server.

    Async so main_judge.py can keep many requests in flight at once: one
    judge processed one request at a time never fills vLLM's continuous
    batching, and throughput on thousands of sequential calls would be
    impractical.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",  # vLLM's OpenAI server does not validate the key
        max_retries: int = 5,
        request_timeout: float = 300.0,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self._model = model
        self._max_retries = max_retries

    def _is_retryable(self, error: Exception) -> bool:
        """4xx client errors (bad request, auth, not-found) will fail the
        same way every time — retrying just wastes the full backoff cycle
        on something that can never succeed. 429 (rate limit) and 5xx/
        network/timeout errors are genuinely transient and worth retrying."""
        try:
            from openai import APIStatusError

            if isinstance(error, APIStatusError):
                return error.status_code == 429 or error.status_code >= 500
        except ImportError:
            pass
        return True  # unknown error type: default to retrying rather than giving up early

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        thinking_budget: int = 0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        # thinking_budget is accepted for interface compatibility with
        # TransformersQwenClient but not used by any of the three judges.
        #
        # 2026-08-28: this file was originally copied verbatim from the
        # extractor's own client (full_test_cot/src/llm_client.py), which
        # hardcoded top_p=0.95, presence_penalty=1.5, and top_k=20 — all
        # specific, sourced recommendations for Qwen3-32B-AWQ in THINKING
        # mode, not generic defaults, and not tuned for Gemma/DeepSeek/
        # Granite at all. Sending presence_penalty=1.5 (an aggressive
        # anti-repetition penalty meant for a known Qwen3 looping issue)
        # to gemma-3-4b-it caused it to return empty completions on 5/5
        # real rows — confirmed by removing it.
        #
        # 2026-08-29: top_p/top_k made optional per-call parameters after
        # checking each judge's own official model card — their
        # recommended values genuinely differ (Gemma: temp=1.0/top_p=0.95/
        # top_k=64; DeepSeek-R1-Distill: temp=0.6/top_p=0.95; Granite-4.x
        # family: temp=1.0/top_p=0.95) — a single hardcoded pair across
        # all three judges was never correct for more than one of them.
        # Left as None (omitted from the request, vLLM/model defaults
        # apply) when a judge's own card gives no specific value.
        kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["extra_body"] = {"top_k": top_k}  # not a standard OpenAI param, vLLM accepts it via extra_body

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content
                if not content:
                    # 2026-08-28: changed from "if content is None" after a
                    # real failure on gemma-3-4b-it — vLLM's own log showed
                    # substantial generation (319.5 tokens/s across 5
                    # concurrent requests, hundreds of tokens each), yet
                    # `content` came back as "" (empty string, not None),
                    # so the old None-only check never triggered and the
                    # real generated text — sitting in reasoning_content —
                    # was silently dropped. "not content" catches both
                    # None and "".
                    content = getattr(message, "reasoning_content", None) or ""
                    logger.warning("content was empty, falling back to reasoning_content (%d chars).", len(content))
                    if not content:
                        # 2026-08-30: added after a real case where BOTH
                        # content and reasoning_content came back empty
                        # despite vLLM's own server log confirming
                        # substantial real generation happened (throughput
                        # stats, not an instant/empty completion). Dumping
                        # the full choice object is the only way to see
                        # where the text actually went — finish_reason,
                        # any other populated field — rather than guessing
                        # at a third fallback field blind.
                        try:
                            full_choice = response.choices[0].model_dump()
                        except Exception:
                            full_choice = repr(response.choices[0])
                        logger.warning("reasoning_content also empty — full choice object: %s", full_choice)
                return content
            except Exception as error:
                last_error = error
                if not self._is_retryable(error) or attempt == self._max_retries - 1:
                    raise
                # Capped exponential backoff with jitter: uncapped growth
                # risks minutes-long waits if max_retries is ever raised,
                # and jitter avoids many concurrent requests all retrying
                # in lockstep after a shared blip (e.g. a server restart).
                wait = min(2**attempt, _MAX_BACKOFF_SECONDS) * (1 + random.random() * 0.25)
                logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                                attempt + 1, self._max_retries, error, wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self._max_retries} attempts") from last_error
