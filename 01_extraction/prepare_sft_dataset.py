"""Converts raw extraction output (.jsonl) into a chat-format SFT dataset.

Why this filters some rows out, not just reformats everything: the
extraction pipeline sometimes has to recover `reasoning` from the model's
native <think> block instead of its disciplined visible answer (see
extractor.py's reasoning_source field). That recovered text is genuine
reasoning, but it never went through the formatting rules the student is
meant to learn (concise, no meta-commentary, consistent style) — training
on it risks teaching the student the same shortcut (skip straight to the
answer) rather than the CoT behavior this whole project exists to distill.

Rows are handled by reasoning_source:
- "visible" (the normal, intended case): always included.
- "native_thinking_fallback": excluded by default. Pass --include-fallback
  to keep them anyway (e.g. if you need the extra volume and are willing to
  accept some stylistic inconsistency in the training data).
- "none" (no reasoning available at all, both channels empty): always
  excluded — there is nothing to train on.

Usage:
    python3 prepare_sft_dataset.py outputs/free_cot_0_10.jsonl sft_ready.jsonl
    python3 prepare_sft_dataset.py outputs/free_cot_0_45889.jsonl sft_ready.jsonl --include-fallback

Output format: one JSON object per line, {"id": ..., "messages": [...]} —
the "messages" list (user prompt, assistant response) matches the
conversational SFT format expected by common fine-tuning frameworks (HF
TRL's SFTTrainer, Axolotl, LLaMA-Factory, unsloth). "id" is kept only for
tracing an example back to its source row, not used in training — strip it
if your framework rejects extra keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prompts import FreeCoTPromptBuilder


def build_assistant_content(row: dict) -> str:
    """Reconstructs a clean, canonical assistant response from the parsed
    fields, rather than reusing raw_response — which may carry formatting
    artifacts (stray whitespace, an inconsistent fence style, the parser's
    balanced-brace fallback trimming) that have nothing to do with what the
    student should actually learn to produce."""
    payload = {
        "schema": row["schema"],
        "entities": [{"text": e["text"], "label": e["label"], "reason": e["reason"]} for e in row["entities"]],
        "excluded": [{"text": e["text"], "reason": e["reason"]} for e in row["excluded"]],
    }
    json_block = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"{row['reasoning']}\n\n```json\n{json_block}\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=str, help="Raw extraction .jsonl (e.g. outputs/free_cot_0_10.jsonl)")
    parser.add_argument("output_path", type=str, help="Where to write the SFT-ready .jsonl")
    parser.add_argument("--include-fallback", action="store_true",
                         help="Also include rows where reasoning was recovered from native_thinking "
                              "instead of the disciplined visible answer. Off by default — see module docstring.")
    args = parser.parse_args()

    prompt_builder = FreeCoTPromptBuilder()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise SystemExit(f"{input_path} not found.")

    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]

    kept, kept_fallback, skipped_none, skipped_fallback = 0, 0, 0, 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            source = row.get("reasoning_source", "visible")  # older shards predate this field
            if source == "none" or not row.get("reasoning"):
                skipped_none += 1
                continue
            if source == "native_thinking_fallback":
                if not args.include_fallback:
                    skipped_fallback += 1
                    continue
                kept_fallback += 1

            example = {
                "id": row["id"],
                "messages": [
                    {"role": "user", "content": prompt_builder.build(row["span"])},
                    {"role": "assistant", "content": build_assistant_content(row)},
                ],
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{kept} examples written to {output_path} ({kept_fallback} via native_thinking fallback)")
    print(f"{skipped_none} skipped (no usable reasoning at all)")
    if args.include_fallback:
        print(f"0 skipped for being fallback (--include-fallback was passed)")
    else:
        print(f"{skipped_fallback} skipped (native_thinking fallback — pass --include-fallback to keep them)")


if __name__ == "__main__":
    main()