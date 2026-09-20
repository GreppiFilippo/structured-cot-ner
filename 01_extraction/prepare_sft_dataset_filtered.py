"""Combines the raw CoT extraction output with the jury's aggregated
verdict, and produces the final SFT-ready dataset — same output shape
as prepare_sft_dataset.py, plus an additional quality filter on the
jury's mean completeness/correctness scores.

Reuses prepare_sft_dataset.py's own FreeCoTPromptBuilder and
build_assistant_content directly (same prompt, same canonical
assistant-response reconstruction) — this script only adds the jury
filtering layer on top, it does not reimplement anything already
correct there.

A row is kept only if it passes BOTH filters:
  1. reasoning_source == "visible" (same as prepare_sft_dataset.py's
     own default — set --include-fallback to relax this too)
  2. jury verdict: mean_completeness >= threshold AND
     mean_correctness >= threshold AND not flagged (high judge
     disagreement) — unless --keep-flagged is passed

Usage:
    python3 prepare_sft_dataset_filtered.py \\
        outputs/free_cot_merged.jsonl \\
        outputs/jury_scores_aggregated.jsonl \\
        sft_ready_filtered.jsonl

    python3 prepare_sft_dataset_filtered.py \\
        outputs/free_cot_merged.jsonl \\
        outputs/jury_scores_aggregated.jsonl \\
        sft_ready_filtered.jsonl \\
        --min-completeness 0.8 --min-correctness 0.8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prompts import FreeCoTPromptBuilder

# reused directly from prepare_sft_dataset.py — not reimplemented
from prepare_sft_dataset import build_assistant_content


def load_jury_verdicts(path: Path) -> dict[str, dict]:
    verdicts = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [warn] {path}:{line_no} is not valid JSON, skipping")
            continue
        verdicts[row["id"]] = row
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=str, help="Raw extraction .jsonl (e.g. outputs/free_cot_merged.jsonl)")
    parser.add_argument("jury_scores_path", type=str,
                         help="Aggregated jury verdicts, from aggregate_jury_scores.py "
                              "(e.g. outputs/jury_scores_aggregated.jsonl)")
    parser.add_argument("output_path", type=str, help="Where to write the filtered, SFT-ready .jsonl")
    parser.add_argument("--min-completeness", type=float, default=0.7,
                         help="Minimum mean_completeness to keep a row (default: 0.7)")
    parser.add_argument("--min-correctness", type=float, default=0.7,
                         help="Minimum mean_correctness to keep a row (default: 0.7)")
    parser.add_argument("--keep-flagged", action="store_true",
                         help="Also keep rows flagged for high judge disagreement "
                              "(off by default — flagged rows are excluded)")
    parser.add_argument("--include-fallback", action="store_true",
                         help="Also include rows where reasoning was recovered from native_thinking "
                              "instead of the disciplined visible answer (same meaning as in "
                              "prepare_sft_dataset.py — off by default)")
    parser.add_argument("--require-jury-verdict", action="store_true", default=True,
                         help="Skip rows with no jury verdict at all (default: True — a row the jury "
                              "never scored has no quality signal to filter on, so it's excluded "
                              "rather than silently assumed good)")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    jury_scores_path = Path(args.jury_scores_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise SystemExit(f"{input_path} not found.")
    if not jury_scores_path.exists():
        raise SystemExit(f"{jury_scores_path} not found. Run aggregate_jury_scores.py first.")

    print(f"Loading jury verdicts from {jury_scores_path}...")
    verdicts = load_jury_verdicts(jury_scores_path)
    print(f"  {len(verdicts)} verdicts loaded.")

    prompt_builder = FreeCoTPromptBuilder()
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"{len(rows)} raw extraction rows loaded from {input_path}.")

    kept = 0
    skip_no_reasoning = 0
    skip_no_verdict = 0
    skip_below_threshold = 0
    skip_flagged = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            source = row.get("reasoning_source", "visible")
            if source == "none" or not row.get("reasoning"):
                skip_no_reasoning += 1
                continue
            if source == "native_thinking_fallback" and not args.include_fallback:
                skip_no_reasoning += 1
                continue

            verdict = verdicts.get(row["id"])
            if verdict is None:
                if args.require_jury_verdict:
                    skip_no_verdict += 1
                    continue
            else:
                if verdict["flagged"] and not args.keep_flagged:
                    skip_flagged += 1
                    continue
                if (verdict["mean_completeness"] < args.min_completeness
                        or verdict["mean_correctness"] < args.min_correctness):
                    skip_below_threshold += 1
                    continue

            example = {
                "id": row["id"],
                "messages": [
                    {"role": "user", "content": prompt_builder.build(row["span"])},
                    {"role": "assistant", "content": build_assistant_content(row)},
                ],
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\n{kept} examples written to {output_path}")
    print(f"  {skip_no_reasoning} skipped (no usable reasoning / fallback excluded)")
    print(f"  {skip_no_verdict} skipped (no jury verdict at all)")
    print(f"  {skip_below_threshold} skipped (below quality threshold: "
          f"completeness>={args.min_completeness}, correctness>={args.min_correctness})")
    print(f"  {skip_flagged} skipped (flagged for high judge disagreement)")


if __name__ == "__main__":
    main()
