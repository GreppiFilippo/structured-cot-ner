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

2026-09-24: that 0.00 pattern did NOT hold on the full 43k-row run —
Granite's completeness varies normally there and correlates best of the
three judges with objective gold coverage (see README), so it should not
be excluded. Until this date the flag also never matched anything (the
files store the full HF id, not the short name), so the actual SFT
selection used all three judges on both axes.

Usage:
    python3 aggregate_jury_scores.py outputs/jury_scores_*.jsonl
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
    n_parse_failed_total = 0
    for path in paths:
        n_rows = 0
        n_parse_failed = 0
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] {path}:{line_no} is not valid JSON, skipping")
                continue
            # 2026-09-02: rows the judge itself failed to parse (from
            # merge_judge_parts.py, which keeps them by default as
            # {"id":..., "parse_failed": true, "raw_response":...} —
            # deliberately, so they're visible for review) have neither
            # completeness_score nor correctness_score. Treat them as
            # "this judge produced no usable verdict for this row" —
            # same as the judge never having scored it at all — rather
            # than crashing the whole aggregation on a missing key.
            if row.get("parse_failed") or "completeness_score" not in row or "correctness_score" not in row:
                n_parse_failed += 1
                continue
            by_id.setdefault(row["id"], []).append(row)
            n_rows += 1
        n_parse_failed_total += n_parse_failed
        print(f"{path}: {n_rows} scores loaded"
              + (f" ({n_parse_failed} parse-failed rows skipped)" if n_parse_failed else ""))
    if n_parse_failed_total:
        print(f"[info] {n_parse_failed_total} parse-failed rows skipped in total across all files "
              f"— those rows are treated as unscored by that judge, not excluded from the dataset entirely "
              f"(they may still have valid verdicts from the other judges).")
    return by_id


def judge_key(judge_model: str) -> str:
    """Normalizes a judge name so every spelling of the same judge compares
    equal: main_judge.py stores the full HF id ("ibm-granite/granite-4.1-8b"),
    the Colab/Kaggle notebooks a short display name ("granite-4.1-8b"), and
    output filenames replace dots with dashes ("granite-4-1-8b")."""
    return judge_model.split("/")[-1].replace(".", "-").lower()


def aggregate(
    by_id: dict[str, list[dict]], disagreement_threshold: float, exclude_completeness_from: set[str]
) -> list[AggregatedScore]:
    excluded_keys = {judge_key(j) for j in exclude_completeness_from}
    results = []
    for record_id, scores in by_id.items():
        compl_scores = [s for s in scores if judge_key(s["judge_model"]) not in excluded_keys]
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
                              "mean — their correctness_score is still included normally. Full HF id, "
                              "short name or filename form all work (e.g. 'granite-4.1-8b').")
    args = parser.parse_args()

    paths = [Path(p) for p in args.score_paths]
    by_id = load_all_scores(paths)
    exclude_set = set(args.exclude_completeness_from)
    if exclude_set:
        print(f"Excluding completeness_score from: {sorted(exclude_set)} (correctness_score still included)")
        seen_judges = {judge_key(s["judge_model"]) for scores in by_id.values() for s in scores}
        unmatched = {j for j in exclude_set if judge_key(j) not in seen_judges}
        if unmatched:
            raise SystemExit(f"--exclude-completeness-from {sorted(unmatched)} matches none of the judges "
                             f"in the input files ({sorted(seen_judges)}) — nothing would be excluded.")

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
