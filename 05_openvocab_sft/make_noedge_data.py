"""Builds the training files of the structured students without the EDGE field,
the alternative structures of the free vs structured CoT experiment.

Same queries and target answers as the structured student: the EDGE instruction
is removed from the prompt (as eval_paired.py does with --prompt noedge) and the
EDGE line from the reasoning, which keeps its TYPES and MENTIONS lines unchanged.
With --invert, MENTIONS is also moved before TYPES, both in the instructions of
the prompt (--prompt noedge_inv) and in the reasoning; the wording of every
field stays the same, so only their order changes.

Usage (from 05_openvocab_sft/):
    python3 make_noedge_data.py --structured data/pilener_paired_structured_cot.jsonl \\
        --out data/pilener_paired_noedge_cot.jsonl
    python3 make_noedge_data.py --invert --out data/pilener_paired_noedge_inv_cot.jsonl
"""
from __future__ import annotations

import argparse
import json

TYPES_INSTRUCTION = "TYPES: Which of the requested types occur in the text, and which appear to be absent.\n"
MENTIONS_INSTRUCTION = ("MENTIONS: The mentions to extract for the types that occur, "
                        "and any plausible ones you rejected.\n")
EDGE_INSTRUCTION = "EDGE: Ambiguous boundaries or type assignments, and how you resolved them.\n"
THINK_START, THINK_END = "<think>\n", "</think>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structured", default="data/pilener_paired_structured_cot.jsonl")
    ap.add_argument("--out", default="data/pilener_paired_noedge_cot.jsonl")
    ap.add_argument("--invert", action="store_true", help="Put MENTIONS before TYPES")
    args = ap.parse_args()

    n = removed_words = kept_words = 0
    with open(args.structured, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            user, assistant = row["messages"]
            prompt = user["content"]
            assert prompt.count(EDGE_INSTRUCTION) == 1, row["id"]
            prompt = prompt.replace(EDGE_INSTRUCTION, "")
            content = assistant["content"]
            assert content.startswith(THINK_START), row["id"]
            reasoning, answer = content[len(THINK_START):].split(THINK_END, 1)
            fields = reasoning.rstrip("\n").split("\n")
            assert [f.split(":", 1)[0] for f in fields] == ["TYPES", "MENTIONS", "EDGE"], row["id"]
            kept = fields[:2]
            if args.invert:
                assert prompt.count(TYPES_INSTRUCTION + MENTIONS_INSTRUCTION) == 1, row["id"]
                prompt = prompt.replace(TYPES_INSTRUCTION + MENTIONS_INSTRUCTION,
                                        MENTIONS_INSTRUCTION + TYPES_INSTRUCTION)
                kept = kept[::-1]
            reasoning = "\n".join(kept) + "\n"
            row["messages"] = [{"role": "user", "content": prompt},
                               {"role": "assistant", "content": f"{THINK_START}{reasoning}{THINK_END}{answer}"}]
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            removed_words += len(fields[2].split())
            kept_words += len(reasoning.split())
    print(f"{n} examples -> {args.out}; reasoning words per example: kept {kept_words / n:.0f}, "
          f"EDGE removed {removed_words / n:.0f}")


if __name__ == "__main__":
    main()
