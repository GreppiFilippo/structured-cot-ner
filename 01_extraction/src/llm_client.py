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
    async def complete(self, prompt: str, temperature: float, max_tokens: int, thinking_budget: int) -> str:
        raise NotImplementedError


class AsyncOpenAICompatibleClient(LLMClient):
    """Works with any OpenAI-compatible endpoint — here, a local vLLM server.

    Async so main.py can keep many requests in flight at once: a dense 27B
    model processed one request at a time never fills vLLM's continuous
    batching, and throughput on 45k+ sequential calls would be impractical.
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
        self, prompt: str, temperature: float = 0.6, max_tokens: int = 4096, thinking_budget: int = 1200
    ) -> str:
        # thinking_budget is accepted for interface compatibility with
        # TransformersQwenClient but NOT sent to the server here: vLLM's
        # reasoning-budget feature is a recent addition, and vllm==0.8.5
        # (pinned for CUDA 12.4/Qwen3 compatibility — see requirements.txt)
        # predates it. Sending an unrecognized extra_body field risks a
        # validation error on every single request if the server's request
        # schema is strict. max_tokens remains the safety net against
        # runaway generation.
        #
        # 2026-08-05: temperature default corrected from 0.7 to 0.6 —
        # after checking Qwen's own recommended settings.
        # Qwen's official docs (confirmed identically across the HF model
        # card, readthedocs quickstart, and independent vendor summaries)
        # specify Temperature=0.6, TopP=0.95, TopK=20, MinP=0 for THINKING
        # mode (which we use, enable_thinking defaults to True) — 0.7 is
        # the recommendation for NON-thinking mode instead, which we had
        # been using by mistake. top_p/top_k were previously left unset,
        # which happened to already default correctly to the model's own
        # generation_config.json (confirmed via a real server log: "Using
        # default chat sampling params from model: {'temperature': 0.6,
        # 'top_k': 20, 'top_p': 0.95}") — set explicitly here now so this
        # doesn't depend on that file being present/correct on every node.
        # Qwen's docs also explicitly warn against greedy decoding here:
        # "can lead to performance degradation and endless repetitions" —
        # notably the same failure mode observed on ner_7 throughout this
        # project's testing, though temperature alone may not fully
        # explain that case.
        #
        # 2026-08-14: presence_penalty=1.5 added after re-reading the
        # official Qwen3-32B-AWQ model card in full — its Best Practices
        # section states: "you can adjust the presence_penalty parameter
        # between 0 and 2 to reduce endless repetitions. We strongly
        # recommend setting this value to 1.5 for quantized models" —
        # directly applicable now that we're on the AWQ checkpoint. Same
        # "endless repetitions" failure mode named above, from the model's
        # own creators, with a concrete recommended fix we hadn't applied.
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=0.95,
                    presence_penalty=1.5,
                    extra_body={"top_k": 20},  # not a standard OpenAI param, vLLM accepts it via extra_body
                    max_tokens=max_tokens,
                )
                message = response.choices[0].message
                content = message.content
                if content is None:
                    # Defensive fallback: seen in practice on third-party
                    # routers whose own server-side split occasionally
                    # attributes the entire response to reasoning_content,
                    # leaving content empty. Our own vLLM server (no
                    # --reasoning-parser) should never hit this, but it's
                    # cheap insurance against a raw None reaching the parser.
                    content = getattr(message, "reasoning_content", None) or ""
                    logger.warning("content was None, falling back to reasoning_content (%d chars).", len(content))
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
