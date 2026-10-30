"""Results table and paired bootstrap of the free vs structured CoT
experiment (outputs of eval_paired.py).

For every run and set: exact typed F1 (requested types only), the same
without MISC, mean reasoning tokens and lost answers (no </think> or no
parsable JSON). Comparisons are paired: the same queries are resampled
(2,000 times) for both runs; over CrossNER the per-domain differences are
averaged, so each domain weighs the same, as in the reported mean.

Usage:
    python3 compare_paired.py --dir outputs_paired > paired_results.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.metrics import exact_counts

CROSSNER = ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics", "crossner_science"]
SETS = CROSSNER + ["pilener_test"]
P = "qwen3-4b-thinking_"
RUNS = {"base": P + "base_free", "free": P + "sft_free", "structured": P + "sft_structured"}
PAIRS = [("free", "base"), ("structured", "base"), ("free", "structured")]
B = 2000


def load(root: Path, ds: str, run: str) -> dict[str, dict]:
    out = {}
    for line in (root / ds / run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        c = [0, 0, 0]
        for t in r["requested_types"]:
            for i, v in enumerate(exact_counts({t: r["gold"][t]}, {t: r["pred"].get(t, [])})):
                c[i] += v
        out[r["id"]] = {"c": c, "reasoning": r["reasoning_tokens"], "lost": r["status"] in ("no_think_end", "parse_error")}
    return out


def f1(c):
    d = c[..., 1] + c[..., 2]
    return np.where(d > 0, 2 * c[..., 0] / np.where(d > 0, d, 1), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs_paired")
    root = Path(ap.parse_args().dir)
    data = {(name, ds): load(root, ds, run) for name, run in RUNS.items() for ds in SETS}

    print(f"{'set':22}" + "".join(f"{n:>26}" for n in RUNS))
    print(f"{'':22}" + "".join(f"{'F1   reason.tok  lost':>26}" for _ in RUNS))
    means = {n: [] for n in RUNS}
    for ds in SETS:
        cells = []
        for n in RUNS:
            d = data[(n, ds)]
            score = float(f1(np.array([v["c"] for v in d.values()]).sum(0)))
            if ds in CROSSNER:
                means[n].append((score, np.mean([v["reasoning"] for v in d.values()])))
            cells.append(f"{score:.3f} {np.mean([v['reasoning'] for v in d.values()]):>9.0f} "
                         f"{sum(v['lost'] for v in d.values()):>5}")
        print(f"{ds:22}" + "".join(f"{c:>26}" for c in cells))
        if ds == CROSSNER[-1]:
            print(f"{'CrossNER mean':22}" + "".join(
                f"{np.mean([s for s, _ in means[n]]):.3f} {np.mean([t for _, t in means[n]]):>9.0f} {'':>5}"
                .rjust(26) for n in RUNS))

    print(f"\nPaired bootstrap ({B} resamples, seed 0): difference of micro F1, 95% interval")
    for group, sets in (("CrossNER mean", CROSSNER), ("pilener_test", ["pilener_test"])):
        for a, b in PAIRS:
            rng = np.random.default_rng(0)
            diffs, point = [], []
            for ds in sets:
                da, db = data[(a, ds)], data[(b, ds)]
                keys = sorted(set(da) & set(db))
                A = np.array([da[k]["c"] for k in keys], float)
                Bm = np.array([db[k]["c"] for k in keys], float)
                idx = rng.integers(0, len(keys), (B, len(keys)))
                diffs.append(f1(A[idx].sum(1)) - f1(Bm[idx].sum(1)))
                point.append(f1(A.sum(0)) - f1(Bm.sum(0)))
            D = np.mean(diffs, 0)
            lo, hi = np.percentile(D, [2.5, 97.5])
            sig = "*" if lo > 0 or hi < 0 else "n.s."
            print(f"  {group:14} {a:>10} - {b:<10} {np.mean(point):+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")


if __name__ == "__main__":
    main()
