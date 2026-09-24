"""Re-scores existing evaluation runs from their predictions.jsonl, without
loading any model or regenerating anything — CPU only, a few seconds per
(model, dataset) pair.

Use it after a change to the parsing/metric code in src/metrics.py (see
METRIC_VERSION there), so every run in outputs/ is scored by exactly the
same code, whenever it was generated. For each
outputs/<dataset>/<pre_sft|post_sft|post_sft_cot>/ folder:
  - re-parses every pred_raw with the current parser,
  - re-selects the gold (for post_sft_cot: the subset's "all_gold"
    column, loaded from the Hub),
  - rewrites metrics.json, per_entity_metrics.json, predictions.jsonl
    and worst_examples.md; the previous metrics.json is kept once as
    metrics_before_rescore.json for comparison.

Usage:
    python3 rescore.py
    python3 rescore.py --datasets ncbi bc5cdr --conditions post_sft_cot
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from src.metrics import CoTPredictionParser, PredictionParser, Scorer, select_gold, write_results

DATASETS = ["ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"]
CONDITIONS = {"pre_sft": "baseline", "post_sft": "pretrain", "post_sft_cot": "cot"}
SUBSET_REPO_ID_DEFAULT = "filippogreppi/bioner-eval-subset"


def load_subset(repo_id, dataset_name, hf_token, subset_dir=None):
    import datasets

    if subset_dir:
        return datasets.load_from_disk(f"{subset_dir}/{dataset_name}")
    return datasets.load_dataset(repo_id, data_files={"test": f"{dataset_name}/test-*"}, token=hf_token)["test"]


def rescore_folder(folder, model_type, subset):
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    parser = CoTPredictionParser if model_type == "cot" else PredictionParser

    scorer = Scorer()
    gold_sources = set()
    new_records = []
    for rec in records:
        if subset is not None:
            gold, gold_source = select_gold(subset[rec["idx"]], model_type)
        else:
            # closed-schema gold is exactly what evaluate.py stored in "gt"
            gold, gold_source = rec["gt"], "prompt_types"
        gold_sources.add(gold_source)
        pred, status = parser.parse(rec["pred_raw"])
        f1, f1_ta = scorer.add(gold, pred, status)
        new_records.append({"idx": rec["idx"], "f1": f1, "f1_type_agnostic": f1_ta, "parse_status": status,
                            "gt": gold, "pred_parsed": pred, "pred_raw": rec["pred_raw"]})

    backup = folder / "metrics_before_rescore.json"
    if (folder / "metrics.json").exists() and not backup.exists():
        shutil.copy(folder / "metrics.json", backup)
    return write_results(folder, scorer, new_records, extra_metrics={"gold_source": sorted(gold_sources)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--subset-repo-id", default=SUBSET_REPO_ID_DEFAULT)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--subset-dir", default=None,
                         help="Read the subset from a local build_eval_subset.py --save-dir folder instead of the Hub")
    args = parser.parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    for ds_name in args.datasets:
        for folder_name in args.conditions:
            folder = Path(args.output_dir) / ds_name / folder_name
            if not (folder / "predictions.jsonl").exists():
                print(f"[skip] {folder}: no predictions.jsonl")
                continue
            model_type = CONDITIONS[folder_name]
            subset = None
            if model_type == "cot":
                subset = load_subset(args.subset_repo_id, ds_name, hf_token, args.subset_dir)
                if "all_gold" not in subset.column_names:
                    raise SystemExit(f"{args.subset_repo_id}/{ds_name} has no 'all_gold' column — rebuild the "
                                     f"subset with build_eval_subset.py before re-scoring the CoT runs.")

            old = {}
            if (folder / "metrics.json").exists():
                old = json.loads((folder / "metrics.json").read_text())
            new = rescore_folder(folder, model_type, subset)
            print(f"{ds_name:<17} {folder_name:<13} "
                  f"F1_ta {old.get('F1_micro_type_agnostic', float('nan')):.4f} -> {new['F1_micro_type_agnostic']:.4f}   "
                  f"F1 {old.get('F1_micro', float('nan')):.4f} -> {new['F1_micro']:.4f}   "
                  f"parse={new['parse_status_counts']}")


if __name__ == "__main__":
    main()
