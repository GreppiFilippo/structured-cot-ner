"""Aggregates per-judge score files (one per judge, from main_judge.py)
into a single per-row verdict: mean completeness/correctness across
judges, and a disagreement flag when judges diverge too much to trust
blindly.

Same logic as jury_colab.ipynb's Jury.aggregate() — reused here at
production scale, reading from files instead of an in-memory list.

2026-08-16: added --exclude-completeness-from. Granite-4.1-8B's
completeness_score came back as exactly 0.00 on 15/15 rows across two
separate real test rounds (37 total judge scores, then another 15),
including after ruling out a tokenization difference from IBM's own
documented usage pattern (see jury_colab.ipynb's use_two_step_tokenize).
Zero variance across genuinely different texts carries no signal to
average in — but its correctness_score varies normally (0.0/1.0 mixed),
so that axis is still included. Not a general-purpose per-judge filter for
arbitrary distrust — only for a score axis with a confirmed, tested,
reproducible lack of variance.

Usage:
    python3 aggregate_jury_scores.py outputs/jury_scores_*.jsonl
    python3 aggregate_jury_scores.py outputs/jury_scores_*.jsonl \
        --exclude-completeness-from granite-4.1-8b
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AggregatedScore:
    id: str
    n_judges_completeness: int
    n_judges_correctness: int
    mean_completeness: float
    mean_correctness: float
    stdev_completeness: float
    stdev_correctness: float
    flagged: bool  # high disagreement between judges — review by hand


def load_all_scores(paths: list[Path]) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    for path in paths:
        n_rows = 0
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] {path}:{line_no} is not valid JSON, skipping")
                continue
            by_id.setdefault(row["id"], []).append(row)
            n_rows += 1
        print(f"{path}: {n_rows} scores loaded")
    return by_id


def aggregate(
    by_id: dict[str, list[dict]], disagreement_threshold: float, exclude_completeness_from: set[str]
) -> list[AggregatedScore]:
    results = []
    for record_id, scores in by_id.items():
        compl_scores = [s for s in scores if s["judge_model"] not in exclude_completeness_from]
        compl = [s["completeness_score"] for s in compl_scores]
        correct = [s["correctness_score"] for s in scores]  # never filtered — correctness is unaffected
        stdev_compl = statistics.stdev(compl) if len(compl) > 1 else 0.0
        stdev_correct = statistics.stdev(correct) if len(correct) > 1 else 0.0
        results.append(AggregatedScore(
            id=record_id,
            n_judges_completeness=len(compl),
            n_judges_correctness=len(correct),
            mean_completeness=statistics.mean(compl) if compl else 0.0,
            mean_correctness=statistics.mean(correct),
            stdev_completeness=stdev_compl,
            stdev_correctness=stdev_correct,
            flagged=stdev_compl > disagreement_threshold or stdev_correct > disagreement_threshold,
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("score_paths", nargs="+", help="jury_scores_<judge>.jsonl file(s) to aggregate")
    parser.add_argument("--output-path", type=str, default="outputs/jury_scores_aggregated.jsonl")
    parser.add_argument("--disagreement-threshold", type=float, default=0.3)
    parser.add_argument("--exclude-completeness-from", nargs="*", default=[],
                         help="judge_model name(s) whose completeness_score should be dropped from the "
                              "mean — their correctness_score is still included normally. Match the "
                              "display_name used when that judge ran (e.g. 'granite-4.1-8b').")
    args = parser.parse_args()

    paths = [Path(p) for p in args.score_paths]
    by_id = load_all_scores(paths)
    exclude_set = set(args.exclude_completeness_from)
    if exclude_set:
        print(f"Excluding completeness_score from: {sorted(exclude_set)} (correctness_score still included)")

    n_missing_judges = sum(1 for scores in by_id.values() if len(scores) < len(paths))
    if n_missing_judges:
        print(f"\n[warn] {n_missing_judges} rows have fewer than {len(paths)} judge scores "
              f"(a judge run may still be in progress, or failed on that row)")

    results = aggregate(by_id, args.disagreement_threshold, exclude_set)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    n_flagged = sum(1 for r in results if r.flagged)
    print(f"\n{len(results)} rows aggregated, written to {output_path}")
    print(f"{n_flagged} rows flagged for high judge disagreement "
          f"({100 * n_flagged / len(results):.1f}%)" if results else "0 rows")


if __name__ == "__main__":
    main()
