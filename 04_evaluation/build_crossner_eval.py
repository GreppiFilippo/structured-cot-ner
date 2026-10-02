"""Builds the CrossNER evaluation subsets (DFKI-SLT/cross_ner, Liu et al. 2021)
in exactly the format of the 9 BioNER subsets (build_eval_subset.py), so
evaluate.py scores them the same way: same system/user prompt, same
requested-types construction (up to MAX_ENTITIES_PER_SAMPLE types: a random
number of the row's present types plus absent distractors drawn with inverse
frequency from the domain's train split), plus raw_text and all_gold.

The five target domains (ai, literature, music, politics, science) are the
general-domain counterpart of the biomedical benchmarks and the standard
zero-shot suite of UniversalNER / InstructUIE / ReasoningNER. Their test
splits are small (416-651 sentences), so each is used IN FULL.

Label handling: CrossNER's labels are lower-case compounds; they are turned
into the evaluation's UPPER_SNAKE style (programlang -> PROGRAMMING_LANGUAGE,
musicalartist -> MUSICAL_ARTIST, ...). "misc" is never requested and kept
out of the gold, as in UniversalNER's benchmark (a catch-all category is not
a meaningful type to ask for by name); its spans stay in all_gold under MISC.

The system prompt is the benchmarks' one verbatim (it calls the task
biomedical): every model was trained and evaluated with it, so the dataset is
the only thing that changes.

Usage:
    python3 build_crossner_eval.py --repo-id yourname/crossner-eval-subset
"""
from __future__ import annotations

import argparse

import datasets
import numpy as np
import sacremoses

from build_eval_subset import MAX_ENTITIES_PER_SAMPLE, RANDOM_SEED, SFTConfig, SFTDataProcessor

DOMAINS = ["ai", "literature", "music", "politics", "science"]
SOURCE = "DFKI-SLT/cross_ner"
REVISION = "refs/convert/parquet"  # the repo only has a loading script; this branch has the data
LABEL_NAMES = {
    "academicjournal": "ACADEMIC_JOURNAL", "astronomicalobject": "ASTRONOMICAL_OBJECT",
    "chemicalcompound": "CHEMICAL_COMPOUND", "chemicalelement": "CHEMICAL_ELEMENT",
    "literarygenre": "LITERARY_GENRE", "musicalartist": "MUSICAL_ARTIST", "musicgenre": "MUSIC_GENRE",
    "musicalinstrument": "MUSICAL_INSTRUMENT", "politicalparty": "POLITICAL_PARTY",
    "programlang": "PROGRAMMING_LANGUAGE",
}


def readable(label: str) -> str:
    return LABEL_NAMES.get(label, label.upper())


def to_string_tags(ds: datasets.Dataset) -> datasets.Dataset:
    """ClassLabel ids -> "B-TYPE"/"I-TYPE"/"O" strings in the evaluation's type names;
    MISC tags become "O" for the requested-type sampling (kept aside for all_gold)."""
    names = ds.features["ner_tags"].feature.names

    def conv(row):
        tags = []
        for t in row["ner_tags"]:
            n = names[t]
            tags.append("O" if n == "O" else f"{n[:2]}{readable(n[2:])}")
        row["str_tags"] = tags
        return row

    return ds.map(conv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=None, help="Private HF dataset repo (one config per domain)")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    detok = sacremoses.MosesDetokenizer(lang="en")
    for dom in DOMAINS:
        raw = datasets.load_dataset(SOURCE, data_dir=dom, revision=REVISION)
        raw = datasets.DatasetDict({k: to_string_tags(v) for k, v in raw.items()})
        # MISC is never requested: drop its tags from the copy the processor sees
        no_misc = raw.map(lambda r: {"tags_no_misc": ["O" if t.endswith("-MISC") else t for t in r["str_tags"]]})
        cfg = SFTConfig(max_entities=MAX_ENTITIES_PER_SAMPLE, random_seed=args.seed, token_field="tokens",
                        tag_field="tags_no_misc", columns_to_remove=("tags_no_misc",))
        proc = SFTDataProcessor(detokenizer=detok, config=cfg).fit(no_misc["train"])
        test = proc.transform(no_misc["test"])
        # all_gold with MISC included (every annotated span of the row)
        from build_eval_subset import IOBSpanExtractor
        import json
        test = test.map(lambda r: {"all_gold": json.dumps(IOBSpanExtractor.extract(r["tokens"], r["str_tags"]),
                                                          ensure_ascii=False)})
        test = test.remove_columns([c for c in test.column_names if c not in ("id", "raw_text", "all_gold", "messages")])
        name = f"crossner_{dom}"
        n_req = [len(json.loads(m[1]["content"].split("ENTITY TYPES:\n")[1])) for m in test["messages"]]
        print(f"{name}: {len(test)} rows, mean requested types {np.mean(n_req):.1f}")
        if args.save_dir:
            test.save_to_disk(f"{args.save_dir}/{name}")
        if args.repo_id:
            test.push_to_hub(args.repo_id, config_name=name, split="test", private=True)
            print(f"  pushed to {args.repo_id} (config {name!r}, private)")


if __name__ == "__main__":
    main()
