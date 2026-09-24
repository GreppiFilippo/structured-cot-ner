"""Paired bootstrap confidence intervals for the F1 differences between
model conditions, per dataset and averaged over the 9 datasets.

Reads only the per-row type-agnostic counts that evaluate.py / rescore.py
store in each predictions.jsonl (metric version 3+), so it needs no model,
no GPU and no network. Rows are resampled with replacement B times; the
same resampled rows are used for both conditions of a comparison (paired),
and for the 9-dataset mean each dataset is resampled independently.

Two metric variants are compared: "all" (every predicted key counts) and
"requested_types_only" (keys that were not among the requested entity
types are ignored; only for models given a list of types — the open-schema
"cot" run always uses "all").

Usage:
    python3 significance.py
    python3 significance.py --n-boot 5000 --csv significance.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DATASETS = ["ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"]
CONDITIONS = {"pre_sft": "Baseline", "post_sft": "Label-only", "post_sft_cot": "CoT", "post_sft_cot_typed": "CoT-typed"}
COMPARISONS = [("post_sft", "pre_sft"), ("post_sft_cot", "pre_sft"), ("post_sft_cot_typed", "pre_sft"),
               ("post_sft_cot", "post_sft"), ("post_sft_cot_typed", "post_sft")]


def load_counts(folder, variant):
    """{idx: [tp, gt, pred]} for one run, or None if missing/not scored with v3."""
    path = folder / "predictions.jsonl"
    if not path.exists():
        return None
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = "counts_type_agnostic" + ("" if variant == "all" else "_requested_types_only")
        if key not in rec:
            key = "counts_type_agnostic"  # open-schema run: no requested types
        if key not in rec:
            return None
        counts[rec["idx"]] = rec[key]
    return counts


def f1(c):
    d = c[..., 1] + c[..., 2]
    return np.where(d > 0, 2 * c[..., 0] / np.where(d > 0, d, 1), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default="outputs/significance.csv")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    out_rows = []
    for variant in ["all", "requested_types_only"]:
        print(f"\n### metric: F1 type-agnostic, {variant}  (Δ, 95% CI; * = CI excludes 0)")
        for a, b in COMPARISONS:
            per_ds, boots = [], []
            for ds in DATASETS:
                ca = load_counts(Path(args.output_dir) / ds / a, variant)
                cb = load_counts(Path(args.output_dir) / ds / b, variant)
                if ca is None or cb is None:
                    continue
                idx = sorted(set(ca) & set(cb))
                A = np.array([ca[i] for i in idx], dtype=float)
                B = np.array([cb[i] for i in idx], dtype=float)
                sample = rng.integers(0, len(idx), (args.n_boot, len(idx)))
                diff_boot = f1(A[sample].sum(1)) - f1(B[sample].sum(1))
                diff = float(f1(A.sum(0)) - f1(B.sum(0)))
                lo, hi = np.percentile(diff_boot, [2.5, 97.5])
                per_ds.append((ds, diff, lo, hi))
                boots.append(diff_boot)
            if not per_ds:
                continue
            name = f"{CONDITIONS[a]} − {CONDITIONS[b]}"
            cells = [f"{ds} {d:+.3f}{'*' if lo > 0 or hi < 0 else ''}" for ds, d, lo, hi in per_ds]
            for ds, d, lo, hi in per_ds:
                out_rows.append({"metric": variant, "comparison": name, "dataset": ds,
                                 "delta": round(d, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
            if len(per_ds) == len(DATASETS):
                mean_boot = np.mean(boots, axis=0)
                mean = float(np.mean([d for _, d, _, _ in per_ds]))
                lo, hi = np.percentile(mean_boot, [2.5, 97.5])
                out_rows.append({"metric": variant, "comparison": name, "dataset": "MEAN",
                                 "delta": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
                head = f"mean {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"
            else:
                head = f"({len(per_ds)}/9 datasets, no mean)"
            print(f"{name:<24} {head}\n{'':<24} " + "  ".join(cells))

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "comparison", "dataset", "delta", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
