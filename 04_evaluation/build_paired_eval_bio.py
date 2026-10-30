"""Builds the biomedical evaluation sets of the free vs structured CoT
experiment, with the same query protocol as the CrossNER ones
(build_paired_eval.py), kept in their own folder (paired_eval/data_bio/).

- Sentences: the same 500 test sentences per dataset as the earlier
  evaluations (build_eval_subset.py: same splits, indices drawn with
  default_rng(42)), checked against structured-cot-ner/bioner-eval-subset.
- Requested types: every type present in the sentence's gold, plus 1-3
  distractors (number drawn uniformly per sentence) among the labels of THE
  SAME DATASET that do not occur in the sentence, shuffled; one fixed seed.
  Each dataset is its own domain, as each CrossNER domain is: a label of
  another dataset is not annotated here (NCBI marks diseases only), so a
  correct mention of it would be scored as a false positive. Single-label
  datasets therefore get a distractor only when the sentence has no entity.
- Type names in the UPPER_SNAKE style of the paired-trace dataset (GeneOrGeneProduct ->
  GENE_OR_GENE_PRODUCT, Body Substance -> BODY_SUBSTANCE).
- Text and gold mentions detokenized with Moses, as for CrossNER, so that
  every gold mention is a substring of the text (the earlier subsets joined
  span tokens with spaces, e.g. "IL - 2", which exact match cannot hit).

Usage:
    python3 build_paired_eval_bio.py --out-dir paired_eval/data_bio
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import datasets
import numpy as np
import sacremoses

from build_eval_subset import BIONER_DATASETS, RANDOM_SEED
from build_paired_eval import SEED, prompts, spans

N_PER_DATASET = 500
CHECK_REPO = "structured-cot-ner/bioner-eval-subset"


def upper_snake(label: str) -> str:
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", label)  # GeneOrGeneProduct -> Gene_Or_Gene_Product
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()


def test_split(ds_id: str) -> datasets.Dataset:
    """The test split exactly as build_eval_subset.load_and_prepare sees it
    (every dataset uses split_mode "auto": its own test split)."""
    return datasets.load_dataset(ds_id)["test"]


def build(ds_id: str, detok, rng: random.Random) -> tuple[list[dict], dict]:
    name = ds_id.split("/")[-1]
    raw = datasets.load_dataset(ds_id)
    labels = sorted({upper_snake(t[2:]) for s in raw.values() for r in s for t in r["ner_tags"]
                     if t.startswith("B-")})
    test = raw["test"]
    n = min(N_PER_DATASET, len(test))
    idx = sorted(np.random.default_rng(RANDOM_SEED).choice(len(test), size=n, replace=False).tolist())
    # read the parquet files directly, as build_eval_subset.matches_existing does (the repo card
    # lists the columns of an older version of the subsets)
    reference = datasets.load_dataset(CHECK_REPO, data_files={"test": f"{name}/test-*"})["test"]["raw_text"]
    rows, st = [], {"gold_mentions": 0, "not_in_text": 0, "text_mismatch": 0}
    for k, i in enumerate(idx):
        r = test[i]
        text = detok.detokenize(r["tokens"])
        st["text_mismatch"] += text != reference[k]
        gold: dict[str, list[str]] = {}
        for typ, s, e in spans(r["tokens"], r["ner_tags"]):
            m = detok.detokenize(r["tokens"][s:e])
            st["gold_mentions"] += 1
            st["not_in_text"] += m not in text
            gold.setdefault(upper_snake(typ), [])
            if m not in gold[upper_snake(typ)]:
                gold[upper_snake(typ)].append(m)
        absent = [l for l in labels if l not in gold]
        distractors = rng.sample(absent, min(rng.randint(1, 3), len(absent)))
        types = list(gold) + distractors
        rng.shuffle(types)
        rows.append({"id": f"{name}_{i}", "text": text, "requested_types": types, "distractors": distractors,
                     "gold": {t: gold.get(t, []) for t in types}, **prompts(text, types)})
    st["labels"] = labels
    return rows, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="paired_eval/data_bio")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detok = sacremoses.MosesDetokenizer(lang="en")
    for ds_id, *_ in BIONER_DATASETS:
        name = ds_id.split("/")[-1]
        rows, st = build(ds_id, detok, random.Random(f"{args.seed}-{name}"))
        (out / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        n_types = sum(len(r["requested_types"]) for r in rows) / len(rows)
        n_dis = sum(len(r["distractors"]) for r in rows) / len(rows)
        print(f"{name}: {len(rows)} sentences (text differs from the earlier subset: {st['text_mismatch']}), "
              f"labels {st['labels']}, {n_types:.2f} types/query ({n_dis:.2f} distractors), "
              f"gold mentions not found verbatim: {st['not_in_text']}/{st['gold_mentions']}")


if __name__ == "__main__":
    main()
