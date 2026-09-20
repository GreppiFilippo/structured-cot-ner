from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.llm_client import LLMClient
from src.prompts import FreeCoTPromptBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    reason: str


@dataclass(frozen=True)
class ExcludedSpan:
    text: str
    reason: str


@dataclass(frozen=True)
class ExtractionResult:
    """One free-CoT extraction for a single Pile-NER span."""

    id: str
    span: str
    schema: list[str]
    entities: list[Entity]
    excluded: list[ExcludedSpan]
    reasoning: str
    reasoning_source: str  # "visible" (as designed) | "native_thinking_fallback" | "none"
    native_thinking: str  # the model's own <think> block, if present — not used for training
    raw_response: str
    gold: dict[str, list[str]] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d


class FreeCoTExtractor:
    """Runs Step 1: open-schema entity extraction with free chain-of-thought.

    Does not rely on server-side reasoning/thinking separation (e.g. vLLM's
    --reasoning-parser): testing against a real API showed this behaves
    inconsistently across backends (the JSON fence in the visible answer was
    present in some calls and absent in others). Instead this always
    receives the full raw generation and splits it client-side, which is
    consistent regardless of server configuration.
    """

    _THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _JSON_FENCED = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[tuple[dict, int]]:
        """Last-resort fallback: scans for the first balanced {...} block
        anywhere in the text and tries to parse it. Handles malformed or
        partial code fences (e.g. an opening ``` that is never closed, or
        missing the 'json' language tag) that neither the strict fenced
        match nor a whole-text parse can handle. Returns the payload and
        the character offset where the JSON object starts, so the caller
        can still recover whatever prose came before it."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1]), start
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _safe_construct(cls, items: object, record_id: str, kind: str) -> list:
        """Builds a list of `cls` (Entity or ExcludedSpan) instances, one
        malformed entry at a time rather than all-or-nothing: an unexpected
        extra key, a missing key, or a non-dict item skips just that one
        entry (logged) instead of raising and discarding the whole record —
        including every other, perfectly valid entity or exclusion in it."""
        if not isinstance(items, list):
            logger.warning("%s: '%s' is not a list (%r), treating as empty.", record_id, kind, type(items))
            return []
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        results = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("%s: skipping non-dict %s entry: %r", record_id, kind, item)
                continue
            filtered = {k: v for k, v in item.items() if k in valid_fields}
            for missing_field in valid_fields - filtered.keys():
                filtered[missing_field] = ""  # default rather than reject the whole entry
            try:
                results.append(cls(**filtered))
            except TypeError as error:
                logger.warning("%s: skipping malformed %s entry %r: %s", record_id, kind, item, error)
        return results

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: Optional[FreeCoTPromptBuilder] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking_budget: int = 1200,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder or FreeCoTPromptBuilder()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

    async def extract(self, record_id: str, span: str, gold: dict[str, list[str]]) -> ExtractionResult:
        prompt = self._prompt_builder.build(span)
        raw = await self._client.complete(
            prompt, temperature=self._temperature, max_tokens=self._max_tokens, thinking_budget=self._thinking_budget
        )
        return self._parse(record_id, span, gold, raw)

    def _parse(self, record_id: str, span: str, gold: dict, raw: str) -> ExtractionResult:
        think_match = self._THINK_BLOCK.search(raw)
        native_thinking = think_match.group(1).strip() if think_match else ""
        visible = self._THINK_BLOCK.sub("", raw).strip()

        fenced_match = self._JSON_FENCED.search(visible)
        if fenced_match:
            payload = json.loads(fenced_match.group(1))
            reasoning = visible[: fenced_match.start()].strip()
        else:
            try:
                payload = json.loads(visible)
                reasoning = ""
                logger.info("%s: no JSON fence and no visible prose reasoning (native thinking only).", record_id)
            except json.JSONDecodeError:
                # Last resort: malformed/partial fence (e.g. an opening ```
                # never closed, or missing the 'json' tag) — scan for the
                # first balanced {...} block instead of giving up.
                fallback = self._extract_first_json_object(visible)
                if fallback is None:
                    raise ValueError(f"No parseable json found in the model output:\n{visible[:500]}")
                payload, json_start = fallback
                reasoning = visible[:json_start].rstrip("` \n").strip()
                logger.info("%s: recovered JSON via balanced-brace fallback (malformed fence).", record_id)

        if not isinstance(payload, dict):
            raise ValueError(f"Parsed JSON is not an object (got {type(payload).__name__}): {str(payload)[:200]}")

        # Observed on real production output (2026-08-04 smoke test, 4/10
        # rows): the model sometimes skips the visible prose entirely and
        # goes straight to JSON, despite the formatting rule asking it not
        # to. Rather than silently lose that row's reasoning material, fall
        # back to native_thinking — but mark the source explicitly, so
        # downstream (jury, dataset curation) can filter or weight these
        # differently from properly rule-following visible reasoning.
        reasoning_source = "visible"
        if not reasoning and native_thinking:
            reasoning = native_thinking
            reasoning_source = "native_thinking_fallback"
            logger.info("%s: visible reasoning was empty, recovered from native_thinking (%d words).",
                        record_id, len(native_thinking.split()))
        elif not reasoning:
            reasoning_source = "none"
            logger.warning("%s: both visible reasoning and native_thinking are empty.", record_id)

        entities = self._safe_construct(Entity, payload.get("entities", []), record_id, "entities")
        excluded = self._safe_construct(ExcludedSpan, payload.get("excluded", []), record_id, "excluded")
        return ExtractionResult(
            id=record_id,
            span=span,
            schema=payload.get("schema", []),
            entities=entities,
            excluded=excluded,
            reasoning=reasoning,
            reasoning_source=reasoning_source,
            native_thinking=native_thinking,
            raw_response=raw,
            gold=gold,
        )
