"""Merges N per-subset judge output files (e.g. jury_scores_<judge>_part1.jsonl,
_part2.jsonl, ...) — one per SLURM/Kaggle/Colab session, for any of the
three judges — into a single jury_scores_<judge>.jsonl, the same shape
main_judge.py itself produces for a non-sharded run.

Once merged, this file can be combined with the other judges' own
outputs using the ALREADY-EXISTING aggregate_jury_scores.py — no new
aggregation logic needed here, just getting a split-across-sessions
output back into the single-file shape the rest of the pipeline expects.

Usage:
    python3 merge_judge_parts.py jury_scores_gemma-3-4b-it_part*.jsonl \
        --output jury_scores_gemma-3-4b-it.jsonl

    python3 merge_judge_parts.py jury_scores_DeepSeek-R1-Distill-Llama-8B_part*.jsonl \
        --output jury_scores_DeepSeek-R1-Distill-Llama-8B.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part_paths", nargs="+", help="jury_scores_<judge>_part*.jsonl files (glob patterns OK)")
    parser.add_argument("--output", required=True, help="e.g. jury_scores_gemma-3-4b-it.jsonl")
    parser.add_argument("--drop-parse-failures", action="store_true",
                         help="Exclude rows marked parse_failed:true from the merged output "
                              "(default: keep them, so they're visible for review — "
                              "aggregate_jury_scores.py already skips anything that doesn't "
                              "parse into a real JudgeScore).")
    args = parser.parse_args()

    # espande eventuali pattern glob passati come stringa letterale (utile
    # se la shell non li ha gia' espansi, es. su Windows)
    all_paths: list[Path] = []
    for p in args.part_paths:
        matched = glob.glob(p)
        all_paths.extend(Path(m) for m in matched) if matched else all_paths.append(Path(p))

    seen_ids: dict[str, str] = {}  # id -> quale file l'ha scritto per primo
    merged: list[dict] = []
    duplicates = 0
    parse_failures = 0

    for path in sorted(all_paths):
        if not path.exists():
            print(f"  [warn] {path} non trovato, salto")
            continue
        n_from_this_file = 0
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] {path}:{line_no} non e' JSON valido, salto")
                continue

            row_id = row.get("id")
            if row_id is None:
                print(f"  [warn] {path}:{line_no} non ha un campo 'id', salto")
                continue

            if row.get("parse_failed"):
                parse_failures += 1
                if args.drop_parse_failures:
                    continue

            if row_id in seen_ids:
                duplicates += 1
                print(f"  [warn] id {row_id!r} gia' visto in {seen_ids[row_id]}, "
                      f"ignoro il duplicato da {path}")
                continue

            seen_ids[row_id] = str(path)
            merged.append(row)
            n_from_this_file += 1

        print(f"  {path}: {n_from_this_file} righe uniche aggiunte")

    with open(args.output, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(merged)} righe totali scritte in {args.output}")
    if duplicates:
        print(f"  ({duplicates} duplicati ignorati — normale se due subset si sono sovrapposti)")
    if parse_failures:
        note = "esclusi" if args.drop_parse_failures else "inclusi ma marcati parse_failed"
        print(f"  ({parse_failures} righe con parsing fallito, {note})")

    print(f"\nProssimo passo — combina con i file jury_scores_<judge>.jsonl degli altri due giudici:")
    print(f"  python3 aggregate_jury_scores.py jury_scores_<giudice1>.jsonl jury_scores_<giudice2>.jsonl "
          f"jury_scores_<giudice3>.jsonl \\")
    print(f"      --exclude-completeness-from granite-4.1-8b  # se necessario")


if __name__ == "__main__":
    main()
