"""Builds an IN-DOMAIN evaluation set from Pile-NER, in exactly the format of
the 9 BioNER subsets (build_eval_subset.py), so evaluate.py scores it the
same way: same system/user prompt, a list of requested entity types (some
present, some absent distractors), gold = the requested types' spans,
plus raw_text and all_gold.

Rows are sampled only among Pile-NER rows that were NOT candidates for
fine-tuning (every id listed in --exclude-ids-from, i.e. the whole
jury-filtered set, not only the examples actually trained on), so the
fine-tuned models have never seen them. Gold is Pile-NER's own annotation
— the same annotation style the models were fine-tuned on — which is the
point: it measures what fine-tuning taught in its own domain, to compare
with the effect on the biomedical benchmarks.

Type sampling mirrors SFTDataProcessor._convert_sample: up to
--max-entities requested types; a random number of them (0..max) are types
present in the row, the rest absent types drawn with inverse frequency.

Usage:
    python3 build_pilener_eval.py --extractions free_cot_merged.jsonl \\
        --exclude-ids-from train.jsonl --repo-id yourname/pilener-eval-subset
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import datasets
import numpy as np

from build_eval_subset import SYSTEM_PROMPT, USER_TEMPLATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractions", required=True, help="free_cot_merged.jsonl (Pile-NER text + gold)")
    parser.add_argument("--exclude-ids-from", required=True, help=".jsonl whose 'id's must not be sampled")
    parser.add_argument("--repo-id", default=None, help="Private HF dataset repo to push to (config 'pilener')")
    parser.add_argument("--save-dir", default=None, help="Also save locally (datasets.save_to_disk)")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-entities", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    excluded = {json.loads(l)["id"] for l in Path(args.exclude_ids_from).read_text().splitlines() if l.strip()}
    rows = [r for r in map(json.loads, Path(args.extractions).read_text().splitlines()) if r["id"] not in excluded]
    print(f"{len(rows)} Pile-NER rows outside the {len(excluded)} excluded ids")

    type_freq = Counter(t for r in rows for t in r["gold"])
    all_types = np.array(sorted(type_freq), dtype=object)
    inv = np.array([1.0 / type_freq[t] for t in all_types])
    weights = inv / inv.sum()

    rng = np.random.default_rng(args.seed)
    picked = [rows[i] for i in sorted(rng.choice(len(rows), size=args.n, replace=False))]

    records = []
    for idx, row in enumerate(picked):
        r = np.random.default_rng(args.seed + idx)
        gold = {t: list(dict.fromkeys(spans)) for t, spans in row["gold"].items()}
        positives = list(gold)
        n_pos = min(len(positives), int(r.integers(0, args.max_entities + 1)))
        chosen = list(r.choice(positives, size=n_pos, replace=False)) if positives else []
        negatives = []
        for c in r.choice(all_types, size=4 * args.max_entities, p=weights):
            if len(chosen) + len(negatives) >= args.max_entities:
                break
            if c not in gold and c not in negatives:
                negatives.append(c)
        entity_list = [str(t) for t in chosen + negatives]
        r.shuffle(entity_list)
        answer = {t: gold.get(t, []) for t in entity_list}
        records.append({
            "id": row["id"],
            "raw_text": row["span"],
            "all_gold": json.dumps(gold, ensure_ascii=False),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    text=row["span"], entities=json.dumps(entity_list, ensure_ascii=False))},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
            ],
        })

    ds = datasets.Dataset.from_list(records)
    n_present = sum(any(json.loads(r["messages"][2]["content"]).values()) for r in records)
    print(f"{len(ds)} rows; {n_present} request at least one type present in the text")
    if args.save_dir:
        ds.save_to_disk(f"{args.save_dir}/pilener")
    if args.repo_id:
        ds.push_to_hub(args.repo_id, config_name="pilener", split="test", private=True)
        print(f"pushed to {args.repo_id} (config 'pilener', private)")


if __name__ == "__main__":
    main()
