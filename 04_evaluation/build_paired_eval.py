"""Builds the evaluation sets of the free vs structured CoT experiment, with
the query protocol of the second phase:

- CrossNER (ai, literature, music, politics, science; full test splits):
  for every sentence, all the types present in its gold annotation, plus 1-3
  distractors (number drawn uniformly per sentence) chosen among the domain's
  labels that do not occur in the sentence; the type order is shuffled. One
  fixed seed, so every model answers exactly the same queries. MISC is a
  label of the domain like the others, so it is requested when present and
  can be drawn as a distractor (the metric is per type, so it can be left out
  afterwards without new generations).
- Pile-NER test split of structured-cot-ner/pilener-teacher-cot-agreement-v1
  (in-domain), used as it is: its requested types and Pile-NER silver labels.

Every row carries both prompts, built with the exact templates of the
paired-trace dataset (checked to reproduce free_prompt and structured_prompt on
all its 5,822 rows): each model is evaluated with the prompt it was trained
on, the base model with both.

CrossNER sentences are detokenized with Moses, and so is every gold span; a
span that is then not a substring of the sentence is reported (exact match
needs the gold mention to be copyable from the text).

Usage:
    python3 build_paired_eval.py --out-dir paired_eval/data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import datasets
import sacremoses

from build_crossner_eval import DOMAINS, REVISION, SOURCE, readable

SEED = 42
PAIRED_REPO = "structured-cot-ner/pilener-teacher-cot-agreement-v1"
FREE_PROMPT = ("Extract named entities from the text for the given entity types.\n"
               "Return a JSON object whose keys are exactly the given entity types and whose values are lists of "
               "mentions copied verbatim from the text. If a type has no mentions, use an empty list.\n\n"
               "Entity types: {types}\n\nText:\n{text}\n\n")
STRUCTURED_SUFFIX = ("\nUse exactly these fields in your thinking, in this order. Write each field on one line.\n"
                     "TYPES: Which of the requested types occur in the text, and which appear to be absent.\n"
                     "MENTIONS: The mentions to extract for the types that occur, and any plausible ones you rejected.\n"
                     "EDGE: Ambiguous boundaries or type assignments, and how you resolved them.\n\n")


def prompts(text: str, types: list[str]) -> dict:
    free = FREE_PROMPT.format(types=", ".join(types), text=text)
    return {"free_prompt": free, "structured_prompt": free + STRUCTURED_SUFFIX}


def spans(tokens: list[str], tags: list[str]) -> list[tuple[str, int, int]]:
    """(type, start, end) of every IOB span."""
    out, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if cur and not t.startswith("I-"):
            out.append((cur[0], cur[1], i))
            cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return out


def build_crossner(dom: str, detok, rng: random.Random) -> tuple[list[dict], dict]:
    splits = datasets.load_dataset(SOURCE, data_dir=dom, revision=REVISION)
    test = splits["test"]
    names = test.features["ner_tags"].feature.names
    # the ClassLabel lists the labels of all domains: keep the ones annotated in this domain
    labels = sorted({readable(names[t][2:]) for s in splits.values() for r in s for t in r["ner_tags"] if t})
    rows, stats = [], {"sentences": 0, "gold_mentions": 0, "not_in_text": 0}
    for k, r in enumerate(test):
        tags = [names[t] for t in r["ner_tags"]]
        text = detok.detokenize(r["tokens"])
        gold: dict[str, list[str]] = {}
        for typ, s, e in spans(r["tokens"], tags):
            m = detok.detokenize(r["tokens"][s:e])
            stats["gold_mentions"] += 1
            stats["not_in_text"] += m not in text
            gold.setdefault(readable(typ), [])
            if m not in gold[readable(typ)]:
                gold[readable(typ)].append(m)
        absent = [l for l in labels if l not in gold]
        distractors = rng.sample(absent, min(rng.randint(1, 3), len(absent)))
        types = list(gold) + distractors
        rng.shuffle(types)
        answer = {t: gold.get(t, []) for t in types}
        rows.append({"id": f"crossner_{dom}_{k}", "text": text, "requested_types": types,
                     "distractors": distractors, "gold": answer, **prompts(text, types)})
        stats["sentences"] += 1
    stats["labels"] = labels
    return rows, stats


def build_pilener() -> list[dict]:
    rows = []
    for r in datasets.load_dataset(PAIRED_REPO, split="test"):
        p = prompts(r["text"], r["requested_types"])
        assert p["free_prompt"] == r["free_prompt"] and p["structured_prompt"] == r["structured_prompt"]
        gold = json.loads(r["silver_answer"])
        rows.append({"id": r["example_id"], "text": r["text"], "requested_types": r["requested_types"],
                     "distractors": [t for t in r["requested_types"] if not gold.get(t)],
                     "gold": {t: gold.get(t, []) for t in r["requested_types"]}, **p})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="paired_eval/data")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detok = sacremoses.MosesDetokenizer(lang="en")

    for dom in DOMAINS:
        rows, st = build_crossner(dom, detok, random.Random(f"{args.seed}-{dom}"))
        (out / f"crossner_{dom}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        n_types = sum(len(r["requested_types"]) for r in rows) / len(rows)
        n_dis = sum(len(r["distractors"]) for r in rows) / len(rows)
        print(f"crossner_{dom}: {st['sentences']} sentences, {len(st['labels'])} labels, "
              f"{n_types:.2f} types/query ({n_dis:.2f} distractors), "
              f"gold mentions not found verbatim: {st['not_in_text']}/{st['gold_mentions']}")
    rows = build_pilener()
    (out / "pilener_test.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"pilener_test: {len(rows)} queries")


if __name__ == "__main__":
    main()
