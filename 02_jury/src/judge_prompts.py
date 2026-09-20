from __future__ import annotations

import json

JURY_TEMPLATE = """You are a judge evaluating an entity extraction produced by another model, against a reference list of gold entity spans for the same text.

The reference list may use different granularity or labeling conventions than the extraction — do not penalize the extraction for using a different (but reasonable) type label, for splitting or merging spans differently, or for spans nested inside one another. Judge meaning, not exact wording.

Text:
{span}

Gold entity spans (reference only, text spans without type):
{gold_spans}

Extraction's reasoning:
{reasoning}

Extraction's final entities:
{entities_json}

Extraction's excluded candidates:
{excluded_json}

Evaluate two things. COMPLETENESS and CORRECTNESS are independent scores — never let one affect the other. For example, an extraction that correctly identifies every gold entity, plus one additional entity that is NOT well-justified by the text, should score completeness_score=1.0 (every gold span is covered) AND correctness_score below 1.0 (the extra entity is unjustified) — the unjustified extra entity does not lower completeness just because it isn't in the gold list, and a correct extra entity does not raise completeness either. Completeness is only about whether the gold spans are covered; correctness is only about whether the non-gold entities are justified. If the gold span list is empty, completeness_score is always 1.0 — there is nothing to miss, so completeness is trivially satisfied regardless of what the extraction found (that extraction's correctness is still judged normally on its own merits).

1. COMPLETENESS — for each gold span, is it (or a clear paraphrase, or a mention that clearly refers to the same real-world entity, including as part of a larger extracted span) meaningfully covered by the extraction's entities or excluded lists? Judge conceptual closeness, not exact string identity — for example, a gold span for a generic category (e.g. "person") and an extraction using a closely related term for the same referent (e.g. "individual", "human") should count as covered, not missed. Only mark a gold span as missing if no reasonable reader would consider it addressed. Report the fraction covered as completeness_score (0.0 to 1.0). If you notice an entity clearly justified in the reasoning is missing from the JSON entities list, treat it as a real completeness gap (score it accordingly, per the rule above) — but explicitly flag this specific mismatch in completeness_notes, since it may indicate a parsing defect in the extraction itself worth investigating, not just a judgment call by the extractor.

2. CORRECTNESS — for each entity the extraction includes that has no clear counterpart in the gold list, judge whether its stated reason is genuinely grounded in the text (not fabricated, not a vague justification) — not whether it happens to match the gold list, which may itself be incomplete or noisy. Report the fraction of extra entities that are genuinely justified as correctness_score (0.0 to 1.0). If there are no extra entities, correctness_score is 1.0.

Respond with only this JSON, nothing else:
```json
{{
  "completeness_score": 0.0,
  "completeness_notes": "one or two sentences",
  "correctness_score": 0.0,
  "correctness_notes": "one or two sentences"
}}
```"""


def flatten_gold(gold: dict) -> list[str]:
    """gold is {label: [span, span, ...]} in the pipeline's schema — the
    judge only ever sees the text spans, never the original labels, so it
    can't just pattern-match on matching type names."""
    spans = []
    for span_list in gold.values():
        spans.extend(span_list)
    seen = set()
    unique = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def build_jury_prompt(row: dict) -> str:
    gold_spans = flatten_gold(row.get("gold", {}))
    return JURY_TEMPLATE.format(
        span=row["span"],
        gold_spans="\n".join(f"- {g}" for g in gold_spans) or "(none)",
        reasoning=row["reasoning"],
        entities_json=json.dumps(row["entities"], ensure_ascii=False, indent=2),
        excluded_json=json.dumps(row["excluded"], ensure_ascii=False, indent=2),
    )
