"""Aggregates the 18 separate metrics.json files (9 datasets x 2 models)
into a single all_results.json + comparison_table.csv — same format as
the original bioner-evaluation-fix.ipynb notebook produced, so it's a
drop-in comparison against the existing 100-sample eval_bioner results.

Usage:
    python3 aggregate_results.py
    python3 aggregate_results.py --output-dir outputs --results-path all_results.json
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
    parser.add_argument("--output-dir", default="outputs", help="Folder containing <dataset>/pre_sft, post_sft")
    parser.add_argument("--results-path", default="all_results.json")
    parser.add_argument("--comparison-path", default="comparison_table.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_results = {}
    missing = []

    for ds_name in DATASETS:
        all_results[ds_name] = {}
        for model_type, key in [("pre_sft", "baseline"), ("post_sft", "finetuned")]:
            metrics_path = output_dir / ds_name / model_type / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            with open(metrics_path) as f:
                all_results[ds_name][key] = json.load(f)

    if missing:
        print(f"[warn] {len(missing)} expected result file(s) not found (job still running, or failed):")
        for m in missing:
            print(f"  {m}")

    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.results_path}")

    rows = []
    for ds_name, res in all_results.items():
        base = res.get("baseline", {})
        fine = res.get("finetuned", {})
        if not base or not fine:
            continue
        rows.append({
            "Dataset": ds_name,
            "Pre-SFT F1": round(base.get("F1_micro", 0.0), 4),
            "Post-SFT F1": round(fine.get("F1_micro", 0.0), 4),
            "\u0394F1": round(fine.get("F1_micro", 0.0) - base.get("F1_micro", 0.0), 4),
            "Pre-SFT P": round(base.get("precision", 0.0), 4),
            "Post-SFT P": round(fine.get("precision", 0.0), 4),
            "Pre-SFT R": round(base.get("recall", 0.0), 4),
            "Post-SFT R": round(fine.get("recall", 0.0), 4),
            "n_samples": fine.get("n_samples", base.get("n_samples", 0)),
        })

    if rows:
        with open(args.comparison_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.comparison_path} ({len(rows)} datasets with complete results)")
    else:
        print("No complete (baseline + finetuned) results yet — comparison table not written.")


if __name__ == "__main__":
    main()
