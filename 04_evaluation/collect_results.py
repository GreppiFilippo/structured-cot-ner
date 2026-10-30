"""Collects every evaluation run of the thesis into one table, without
touching the original result folders.

For each (run, dataset) found in the result folders it reads
predictions.jsonl and reports, with the current src/metrics.py:
  - the type-agnostic token-IoU F1 used so far (F1_ta), and
  - the exact typed F1 (F1_exact, METRIC_VERSION 4), recomputed from the
    stored gold ("gt") and parsed prediction ("pred_parsed"), so no model
    is run and nothing is re-parsed. Only the requested types (the keys of
    "gt", distractors included) are scored, as in the second phase: keys the
    model adds outside them are ignored. On CrossNER and the biomedical sets
    the gold mentions are first Moses-detokenized like the text the model saw
    (see score_folder).
Open-schema runs (phase 1 "post_sft_cot") name their own types, so the
typed exact metric is not defined for them (left empty).

When the same run appears in several folders, the first source in SOURCES
wins (raw Modal backups before the older local copies).

Usage:
    python3 04_evaluation/collect_results.py --out thesis_results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.metrics import exact_counts, f1_from_counts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# (folder, description); every folder is laid out as <folder>/<dataset>/<run>/
SOURCES = [
    ("modal_terzo_backup/eval", "Modal (account 3), CrossNER completion runs, 2026-10-15"),
    ("modal_secondo_backup/eval", "Modal (account 2): Qwen3-4B v5t, MedGemma, type descriptions"),
    ("modal_primary_backup2/eval", "Modal (account 1): Qwen3-1.7B v5A/B/C jury-threshold runs"),
    ("modal_primary_backup/eval", "Modal (account 1): Qwen3-1.7B v3/v4 series, Qwen3-4B v4/v4lo/base"),
    ("04_evaluation/outputs_modal", "local copy of Modal results (older download)"),
    ("04_evaluation/outputs_qwen", "Windows PC, transformers engine, 4096 tokens"),
    ("04_evaluation/outputs", "phase 1: MedGemma baseline / label-only / open-schema CoT"),
]
ENGINE = {src: "transformers (PC, 4096 tok)" if src.endswith("outputs_qwen")
          else "transformers (phase 1)" if src.endswith("outputs") else "vLLM (Modal, 6144 tok)"
          for src, _ in SOURCES}
PHASE1 = {"pre_sft": "medgemma-1.5-4b-it_phase1_baseline",
          "pre_sft_1536": "medgemma-1.5-4b-it_phase1_baseline_1536tok",
          "post_sft": "medgemma-1.5-4b-it_phase1_label_only",
          "post_sft_cot": "medgemma-1.5-4b-it_phase1_open_schema_cot"}
BIO = ["ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"]
CROSSNER = ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics", "crossner_science"]
SKIP_RUNS = {"smoke_base", "smoke_bosfix", "smoke_eval_base", "smoke_sft"}
# discarded: MedGemma evaluated with a double <bos> token (superseded by the _bosfix runs)
DISCARDED = {"medgemma-1.5-4b-it_base", "medgemma-1.5-4b-it_sft_v5t"}


def group(ds):
    return "biomedical" if ds in BIO else "crossner" if ds in CROSSNER else "pilener" if ds == "pilener" else "other"


def score_folder(folder: Path, open_schema: bool, detok=None):
    """detok: Moses detokenizer for the sets built from BIO tags. Their first-phase
    subsets show the model Moses-detokenized text but stored the gold mentions as
    space-joined tokens ("Graves ' disease" for "Graves' disease"), which exact
    matching cannot hit; the mentions are detokenized like the text, as the
    second-phase sets do (build_paired_eval_bio.py)."""
    tp = gt = pc = 0
    tp_ta = gt_ta = pc_ta = 0.0
    n = parse_err = 0
    out_chars = 0
    for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        parse_err += r.get("parse_status") == "parse_error"
        out_chars += len(r.get("pred_raw", ""))
        c = r.get("counts_type_agnostic")
        if c:
            tp_ta += c[0]; gt_ta += c[1]; pc_ta += c[2]
        if not open_schema:
            # only the requested types are scored (the keys of "gt", distractors included),
            # as in the second phase (eval_paired.score): a key the model invents is ignored
            gold = r.get("gt") or {}
            if detok is not None:
                gold = {k: [detok.detokenize(m.split()) for m in v] for k, v in gold.items()}
            pred = {k: v for k, v in (r.get("pred_parsed") or {}).items() if k in gold}
            a, b, d = exact_counts(gold, pred)
            tp += a; gt += b; pc += d
    row = {
        "n": n, "parse_errors": parse_err,
        "F1_ta": f1_from_counts(tp_ta, gt_ta, pc_ta), "P_ta": tp_ta / pc_ta if pc_ta else 0.0,
        "R_ta": tp_ta / gt_ta if gt_ta else 0.0,
        "mean_output_chars": out_chars / n if n else 0.0,
    }
    if open_schema:
        row.update({"F1_exact": "", "P_exact": "", "R_exact": ""})
    else:
        row.update({"F1_exact": f1_from_counts(tp, gt, pc), "P_exact": tp / pc if pc else 0.0,
                    "R_exact": tp / gt if gt else 0.0})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="thesis_results")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    import sacremoses
    detok = sacremoses.MosesDetokenizer(lang="en")
    rows, seen = [], set()
    for src, desc in SOURCES:
        base = ROOT / src
        if not base.exists():
            continue
        for folder in sorted(base.glob("*/*")):
            if not (folder / "predictions.jsonl").exists():
                continue
            ds, run = folder.parent.name, folder.name
            if src == "04_evaluation/outputs":
                run = PHASE1.get(run, run)
            elif src == "04_evaluation/outputs_qwen":
                run += "_pc_transformers"  # same models as on Modal, different engine
            if run in SKIP_RUNS or (run, ds) in seen:
                continue
            seen.add((run, ds))
            row = {"run": run, "dataset": ds, "group": group(ds), "engine": ENGINE[src],
                   "status": "discarded (double BOS)" if run in DISCARDED else "valid",
                   "source": src, "path": str(folder.relative_to(ROOT))}
            row.update(score_folder(folder, open_schema=folder.name == "post_sft_cot",
                                    detok=detok if ds in BIO or ds in CROSSNER else None))
            rows.append(row)

    rows.sort(key=lambda r: (r["run"], r["group"], r["dataset"]))
    cols = ["run", "dataset", "group", "engine", "status", "F1_exact", "P_exact", "R_exact", "F1_ta", "P_ta", "R_ta",
            "n", "parse_errors", "mean_output_chars", "source", "path"]
    with open(out / "all_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # per-run means per group (only complete groups)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["run"]][r["group"]].append(r)
    size = {"biomedical": 9, "crossner": 5, "pilener": 1}
    with open(out / "summary_by_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "status", "group", "datasets", "mean_F1_exact", "mean_F1_ta", "mean_output_chars"])
        for run in sorted(agg):
            for g, rs in sorted(agg[run].items()):
                if g not in size:
                    continue
                ex = [r["F1_exact"] for r in rs if r["F1_exact"] != ""]
                w.writerow([run, rs[0]["status"], g, f"{len(rs)}/{size[g]}",
                            f"{sum(ex) / len(ex):.4f}" if ex else "",
                            f"{sum(r['F1_ta'] for r in rs) / len(rs):.4f}",
                            f"{sum(r['mean_output_chars'] for r in rs) / len(rs):.0f}"])
    print(f"{len(rows)} (run, dataset) results -> {out / 'all_results.csv'}, {out / 'summary_by_run.csv'}")


if __name__ == "__main__":
    main()
