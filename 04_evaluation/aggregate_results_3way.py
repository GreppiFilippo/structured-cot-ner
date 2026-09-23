"""Aggregates metrics.json files across all three model conditions
(baseline / pre_sft, label-only SFT / post_sft, CoT-distilled /
post_sft_cot) into a single all_results.json + comparison_table.csv —
same format/filenames as the original two-way aggregate_results.py, plus
a third column. Backward compatible: works fine if post_sft_cot doesn't
exist yet for some datasets (those rows just omit the CoT columns).

Usage:
    python3 aggregate_results_3way.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = [
    "ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM",
    "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="outputs",
                         help="Folder containing <dataset>/pre_sft, post_sft, post_sft_cot")
    parser.add_argument("--results-path", default="all_results.json")
    parser.add_argument("--comparison-path", default="comparison_table.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_results = {}
    missing = []

    conditions = [("pre_sft", "baseline"), ("post_sft", "label_only"), ("post_sft_cot", "cot")]

    for ds_name in DATASETS:
        all_results[ds_name] = {}
        for folder_name, key in conditions:
            metrics_path = output_dir / ds_name / folder_name / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            with open(metrics_path) as f:
                all_results[ds_name][key] = json.load(f)

    if missing:
        print(f"[warn] {len(missing)} expected result file(s) not found (job still running, or not launched yet):")
        for m in missing:
            print(f"  {m}")

    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.results_path}")

    rows = []
    for ds_name, res in all_results.items():
        base = res.get("baseline", {})
        label_only = res.get("label_only", {})
        cot = res.get("cot", {})
        if not base:
            continue  # need at least the baseline to build a comparison row
        row = {
            "Dataset": ds_name,
            "Baseline F1": round(base.get("F1_micro", 0.0), 4),
        }
        if label_only:
            row["Label-only SFT F1"] = round(label_only.get("F1_micro", 0.0), 4)
            row["\u0394 Label-only"] = round(label_only.get("F1_micro", 0.0) - base.get("F1_micro", 0.0), 4)
        if cot:
            # 2026-09-22: F1_micro_type_agnostic, not F1_micro — the CoT
            # model is open-schema and invents its own type names, so
            # F1_micro (which compares type names exactly against a
            # fixed closed-schema gold vocabulary) reads near-zero even
            # when the model finds the right spans under a different,
            # semantically valid type name. Confirmed directly: a real
            # run showed 499/500 successful parses and F1_micro=0.0 on
            # ncbi, with F1_micro_type_agnostic=0.31 on the same rows.
            row["CoT-distilled F1"] = round(cot.get("F1_micro_type_agnostic", 0.0), 4)
            row["\u0394 CoT"] = round(cot.get("F1_micro_type_agnostic", 0.0) - base.get("F1_micro", 0.0), 4)
        row["n_samples"] = (cot or label_only or base).get("n_samples", 0)
        rows.append(row)

    if rows:
        # colonne unite di tutte le righe, per gestire dataset con solo
        # alcune condizioni completate senza scartare l'intera riga
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(args.comparison_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.comparison_path} ({len(rows)} datasets with at least a baseline result)")
    else:
        print("No results yet — comparison table not written.")


if __name__ == "__main__":
    main()

