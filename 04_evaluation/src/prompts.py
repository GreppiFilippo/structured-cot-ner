"""Verbatim copy of full_test_cot/src/prompts.py's FreeCoTPromptBuilder —
the EXACT prompt actually baked into the SFT training data (the
teacher-framed opening paragraph was deliberately kept as-is, not
replaced with the cleaned-up student-framed version, per an explicit
decision made during dataset preparation). Copied here rather than
imported across folders, matching this project's established pattern of
each pipeline stage keeping its own self-contained copy of shared code.

Using ANY other prompt to evaluate the CoT-distilled model — including
the closed-schema prompt built by build_eval_subset.py for
baseline/pretrain, or the cleaned-up StudentPromptBuilder — would
evaluate the model on a task it was never trained to expect the
instructions for, artificially depressing its measured performance.
"""
from __future__ import annotations

FREE_COT_TEMPLATE = """You are generating high-quality synthetic reasoning data. This data will be used to train a small model to perform open-schema named entity recognition through explicit reasoning. Your role is the teacher: think through the task carefully and completely, because the small model will learn to imitate exactly the reasoning process you demonstrate, not just the final answer.

You are given a single text. It may come from any domain, language, or format: news, code, legal filings, biomedical papers, social media, configuration files, mathematical problems, poetry, or anything else. You are not given a fixed list of entity types.

Follow this procedure:

1. Read the text once, in order, and judge whether it plausibly contains any named entities at all. Many texts (code, boilerplate, generic math, license headers, abstract discussion) contain none — this is a normal, correct outcome, not a failure. Do not invent entities to avoid an empty result.
2. For every span you consider a candidate entity, ground your judgment in what the text actually says. Prefer evidence stated in the text over assumptions from general knowledge; if you do need outside knowledge to identify what something is (e.g. recognizing an organization name), state that plainly, but never contradict what the text itself says. Never invent supporting evidence that is not actually there — if you claim a term appears elsewhere in the text (a second mention, a matching detail in a code block, an earlier sentence), that claim must be literally true. A conclusion that is already correct on its own never needs fabricated corroboration.
3. Choose entity type labels that are as specific and informative as the context supports, without inventing needless one-off synonyms for the same concept. Two or three illustrations of the specificity wanted: "Nationality" instead of a generic "Location" when the span is a demonym of origin; "ChemicalCompound" instead of a generic "Substance" in a chemistry text; "SportsTeam" instead of a generic "Organization" when the referent is a team. These are illustrations of a level of precision, not a list to copy from — invent whatever label fits the text. This also covers technical domains: a programming language, framework, protocol, or file format named in natural-language discussion about it (e.g. "written in Lua") is a real entity just like a person or organization would be — but a code-only token used purely as syntax (a namespace, a variable name, a library import path) is not, even if it happens to share a name with a real technology.
4. Watch for figurative, idiomatic, or metonymic uses of otherwise literal terms — for example a country name standing for its team, government, or population rather than the landmass itself, or crossing an ocean used as a metaphor for cultural influence rather than a literal voyage. Label the entity according to what it actually refers to in context, and apply that same reading consistently every time the same term recurs with the same sense in the text. If the use is purely rhetorical and does not denote a real, identifiable referent, exclude it.
5. For every span that could plausibly look like an entity but is not one (a generic role, a common noun, a vague quantity, a relative date, a descriptive phrase), say so briefly and explain why — only if such a case is genuinely present. Do not force an exclusion if none stands out. If you do analyze and reject a candidate in your reasoning, it must also appear in the `excluded` array — never discuss a false positive in prose and then leave that array empty.
6. If the same real-world entity is mentioned more than once (full name, then a shortened or pronominal reference), report it once, using its clearest surface form.
7. Strip leading articles ("the", "a", "an") from a span unless they are part of the proper name itself. Keep every surface form in the original language of the text — never translate it.

Formatting rules:
- Reason in plain prose, moving through the text in order. Use as much reasoning as the text's ambiguity genuinely requires — a short, unremarkable text deserves a short analysis; do not pad it, and do not compress a genuinely ambiguous case into a single unjustified line.
- Never discuss the output format, JSON syntax, or how you will write the answer.
- Never restate a conclusion you have already justified.
- `entities` and `excluded` are both arrays that may contain zero, one, or many items — always include both keys even when empty. The example below shows the shape of a single item, not a limit.
- Every conclusion your reasoning reaches about a candidate span — including it as an entity or rejecting it as a false positive — must be reflected in the corresponding JSON array. Never reach a conclusion in prose and then leave it out of the JSON, in either direction.
- Your visible answer is the only part of this response that will ever be used afterward — nothing that happens before it is kept. It must stand completely on its own: write out your reasoning in full inside the visible answer, even if you already worked through the same steps internally beforehand. Do not shorten or skip the visible reasoning on the assumption that it was already covered elsewhere.
- When your reasoning is complete, write nothing else — your very next character must be the opening backtick of the JSON block below.
- Your visible answer must contain only the prose analysis and the JSON block described above — no stray tags, no meta-commentary about your own process, nothing else before, between, or after them.

Before you answer, keep these three in mind — they are the easiest rules to slip on: an empty result is a correct result when the text has no entities; a country, city, or landmark name used to mean its team, government, or population is not the same entity as the place itself; and a candidate you are unsure about needs a real textual reason, never "for simplicity" or similar.

Text:
{span}

```json
{{
  "schema": ["EntityType1", "EntityType2"],
  "entities": [
    {{"text": "exact surface form", "label": "EntityType", "reason": "short evidence-based justification"}}
  ],
  "excluded": [
    {{"text": "surface form", "reason": "short reason it is not a named entity"}}
  ]
}}
```"""


class FreeCoTPromptBuilder:
    def build(self, span: str) -> str:
        return FREE_COT_TEMPLATE.format(span=span)
