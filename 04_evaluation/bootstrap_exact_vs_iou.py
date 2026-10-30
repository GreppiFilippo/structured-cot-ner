"""Paired bootstrap of the main comparisons of the thesis, under both the
type-agnostic token-IoU F1 (metric v3) and the exact typed F1 (metric v4,
recomputed from the stored gt / pred_parsed): checks whether a conclusion
depends on the metric.

For each dataset, sentences are resampled (2,000 times, same indices for
the two runs), micro F1 is computed per run, and the per-dataset differences
are averaged; significance = 95% interval not containing 0.

Usage, from the repository root:
    python3 04_evaluation/bootstrap_exact_vs_iou.py > thesis_results/bootstrap_exact_vs_iou.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sacremoses

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.metrics import exact_counts  # noqa: E402

DETOK = sacremoses.MosesDetokenizer(lang="en")

ROOT = Path(__file__).resolve().parent.parent
# raw Modal results, most recent download first (same priority as collect_results.py)
BASES = ["modal_terzo_backup/eval", "modal_secondo_backup/eval", "modal_primary_backup2/eval",
         "modal_primary_backup/eval"]
CROSSNER = ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics", "crossner_science"]
BIO = ["ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"]
B = 2000
P = "qwen3-1.7b_"
PAIRS = [(P + "sft_v4", P + "sft_v4rej"), (P + "sft_v4", P + "sft_v4all"), (P + "sft_v4all", P + "sft_v4rej"),
         (P + "sft_v4", P + "base"), (P + "sft_v4", P + "sft_v4lo"), (P + "sft_v5A", P + "sft_v5B"),
         (P + "sft_v5C", P + "sft_v5A"), ("qwen3-4b_sft_v5t", "qwen3-4b_sft_v4"), ("qwen3-4b_sft_v5t", "qwen3-4b_base")]


def exact(r):
    """Exact counts as in collect_results.score_folder: requested types only, gold
    mentions Moses-detokenized like the text (all the sets compared here are built
    from BIO tags)."""
    gold = {k: [DETOK.detokenize(m.split()) for m in v] for k, v in (r.get("gt") or {}).items()}
    pred = {k: v for k, v in (r.get("pred_parsed") or {}).items() if k in gold}
    return exact_counts(gold, pred)


def counts(ds, run, kind):
    for b in BASES:
        f = ROOT / b / ds / run / "predictions.jsonl"
        if f.exists():
            out = {}
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[r["idx"]] = r["counts_type_agnostic"] if kind == "iou" else exact(r)
            return out
    raise FileNotFoundError(f"{ds}/{run}")


def f1(c):
    d = c[..., 1] + c[..., 2]
    return np.where(d > 0, 2 * c[..., 0] / np.where(d > 0, d, 1), 1.0)


def compare(a, b, datasets, label, kind):
    rng = np.random.default_rng(0)
    diffs, point = [], []
    for ds in datasets:
        ca, cb = counts(ds, a, kind), counts(ds, b, kind)
        keys = sorted(set(ca) & set(cb))
        A = np.array([ca[k] for k in keys], float)
        Bm = np.array([cb[k] for k in keys], float)
        idx = rng.integers(0, len(keys), (B, len(keys)))
        diffs.append(f1(A[idx].sum(1)) - f1(Bm[idx].sum(1)))
        point.append(f1(A.sum(0)) - f1(Bm.sum(0)))
    D = np.mean(diffs, 0)
    lo, hi = np.percentile(D, [2.5, 97.5])
    sig = "*" if lo > 0 or hi < 0 else "n.s."
    print(f"{label:7}{kind:6}{a:22} vs {b:22} diff={np.mean(point):+.3f} CI95=[{lo:+.3f},{hi:+.3f}] {sig}")


def main() -> None:
    print("Paired bootstrap, 2000 resamples, mean of per-dataset micro-F1 differences (seed 0)\n")
    for name, datasets in (("CrossN", CROSSNER), ("Bio", BIO)):
        for a, b in PAIRS:
            for kind in ("iou", "exact"):
                compare(a, b, datasets, name, kind)
        print()


if __name__ == "__main__":
    main()
