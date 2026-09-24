"""Aggregates metrics.json files across all three model conditions
(baseline / pre_sft, label-only SFT / post_sft, CoT-distilled /
post_sft_cot) into a single all_results.json + comparison_table.csv.

The headline columns (F1 / P / R and the deltas) use the type-agnostic
metric for ALL THREE models: it is the only one that can be computed the
same way for the open-schema CoT model, which names its own entity
types. Comparing the CoT model's type-agnostic F1 against the other two
models' type-exact F1 (as this script did before 2026-09-24) mixes two
different metrics in one delta. Type-exact F1 is still reported for
baseline and label-only, in its own columns.

Rows scored with an older metric version, or CoT rows not scored against
the full gold, are reported with a warning — run rescore.py to update them.

Usage:
    python3 aggregate_results_3way.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.metrics import METRIC_VERSION

DATASETS = [
    "ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM",
    "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored",
]
CONDITIONS = [("pre_sft", "baseline", "Baseline"),
              ("post_sft", "label_only", "Label-only"),
              ("post_sft_cot", "cot", "CoT")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="outputs",
                         help="Folder containing <dataset>/pre_sft, post_sft, post_sft_cot")
    parser.add_argument("--results-path", default="all_results.json")
    parser.add_argument("--comparison-path", default="comparison_table.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_results = {}
    missing, stale = [], []

    for ds_name in DATASETS:
        all_results[ds_name] = {}
        for folder_name, key, _ in CONDITIONS:
            metrics_path = output_dir / ds_name / folder_name / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            metrics = json.loads(metrics_path.read_text())
            all_results[ds_name][key] = metrics
            if metrics.get("metric_version") != METRIC_VERSION:
                stale.append(f"{metrics_path} (metric_version={metrics.get('metric_version', 1)})")
            elif key == "cot" and metrics.get("gold_source") != ["all_gold"]:
                stale.append(f"{metrics_path} (gold_source={metrics.get('gold_source')})")

    if missing:
        print(f"[warn] {len(missing)} expected result file(s) not found (job still running, or not launched yet):")
        for m in missing:
            print(f"  {m}")
    if stale:
        print(f"[warn] {len(stale)} result(s) not scored with the current metric (v{METRIC_VERSION}) "
              f"— run rescore.py, their numbers are not comparable with the rest:")
        for s in stale:
            print(f"  {s}")

    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.results_path}")

    rows = []
    for ds_name, res in all_results.items():
        if not res:
            continue
        row = {"Dataset": ds_name}
        base_f1 = res.get("baseline", {}).get("F1_micro_type_agnostic")
        for _, key, name in CONDITIONS:
            m = res.get(key)
            if not m or "F1_micro_type_agnostic" not in m:
                continue
            row[f"{name} F1"] = round(m["F1_micro_type_agnostic"], 4)
            row[f"{name} P"] = round(m["precision_type_agnostic"], 4)
            row[f"{name} R"] = round(m["recall_type_agnostic"], 4)
            if key != "baseline" and base_f1 is not None:
                row[f"Δ {name}"] = round(m["F1_micro_type_agnostic"] - base_f1, 4)
            if key != "cot":
                row[f"{name} F1 (type-exact)"] = round(m["F1_micro"], 4)
        row["n_samples"] = next(m["n_samples"] for m in res.values())
        rows.append(row)

    if rows:
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(args.comparison_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.comparison_path} ({len(rows)} datasets)")
    else:
        print("No results yet — comparison table not written.")


if __name__ == "__main__":
    main()
