"""Tries every non-empty combination of the three judges (7 total: each
alone, every pair, all three together), and reports how many rows would
pass the SFT quality filter under each combination — to diagnose which
judge(s), if any, are responsible for the low overall yield, rather than
testing combinations one at a time by hand.

Reuses aggregate_jury_scores.py's own load_all_scores() and aggregate()
directly — this script only adds the "try every subset" loop and the
summary table on top, no aggregation logic is reimplemented.

Usage:
    python3 try_judge_combinations.py \\
        outputs/jury_scores_gemma-3-4b-it.jsonl \\
        outputs/jury_scores_granite-4-1-8b.jsonl \\
        outputs/jury_scores_DeepSeek-R1-Distill-Llama-8B.jsonl \\
        --min-completeness 0.6 --min-correctness 0.6
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from aggregate_jury_scores import aggregate, load_all_scores


def short_name(path: Path) -> str:
    """Turns 'outputs/jury_scores_gemma-3-4b-it.jsonl' into 'gemma-3-4b-it'."""
    stem = path.stem
    prefix = "jury_scores_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judge_paths", nargs="+", help="All three raw per-judge jsonl files")
    parser.add_argument("--min-completeness", type=float, default=0.6)
    parser.add_argument("--min-correctness", type=float, default=0.6)
    parser.add_argument("--disagreement-threshold", type=float, default=0.3)
    args = parser.parse_args()

    paths = [Path(p) for p in args.judge_paths]
    names = [short_name(p) for p in paths]

    # carico ciascun file UNA sola volta, poi provo tutte le combinazioni
    # in memoria senza rileggere i file da disco ogni volta
    print("Loading all judge files once...")
    per_judge_scores: dict[str, dict[str, list[dict]]] = {}
    for path, name in zip(paths, names):
        by_id = load_all_scores([path])
        per_judge_scores[name] = by_id
    print()

    results = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            combo_names = [names[i] for i in combo]

            # unisco i verdetti delle sole righe di questa combinazione,
            # stessa struttura per-id attesa da aggregate()
            merged_by_id: dict[str, list[dict]] = {}
            for i in combo:
                for row_id, scores in per_judge_scores[names[i]].items():
                    merged_by_id.setdefault(row_id, []).extend(scores)

            aggregated = aggregate(merged_by_id, args.disagreement_threshold, exclude_completeness_from=set())

            n_pass = sum(
                1 for v in aggregated
                if v.mean_completeness >= args.min_completeness
                and v.mean_correctness >= args.min_correctness
            )
            n_pass_not_flagged = sum(
                1 for v in aggregated
                if v.mean_completeness >= args.min_completeness
                and v.mean_correctness >= args.min_correctness
                and not v.flagged
            )
            results.append({
                "judges": " + ".join(combo_names),
                "n_judges": len(combo_names),
                "n_pass": n_pass,
                "n_pass_not_flagged": n_pass_not_flagged,
            })

    results.sort(key=lambda r: (-r["n_judges"], -r["n_pass"]))

    print(f"{'Judges':<55} {'Rows >= threshold':>18} {'...and not flagged':>20}")
    print("-" * 95)
    for r in results:
        print(f"{r['judges']:<55} {r['n_pass']:>18} {r['n_pass_not_flagged']:>20}")


if __name__ == "__main__":
    main()
