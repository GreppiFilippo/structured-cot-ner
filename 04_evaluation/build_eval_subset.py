"""Builds a FIXED evaluation subset for each of the 9 BioNER datasets and
uploads it as a private multi-split dataset on HF Hub.

Why this exists: reusing the exact same subset (same rows, same generated
prompt) across all three models being compared (baseline, MedGemma-pretrain,
and eventually the CoT-distilled final model) removes a real source of
noise — otherwise each evaluation run would sample a different random
subset AND regenerate the distractor entities in the prompt with a
different random draw, so even the "same" 500 rows could get a subtly
different prompt each time (see SFTDataProcessor._convert_sample: distractor
entities are drawn with a seed tied to the row's index within a call, not
persisted). Saving the fully-built {"messages": [...]} objects (not just
raw text + IOB tags) freezes the exact prompt text permanently, so a
comparison made months from now is still apples-to-apples with what's
being evaluated today.

Reuses IOBSpanExtractor / EntityDistribution / SFTDataProcessor verbatim
from bioner-evaluation-fix.ipynb (the notebook that produced the existing
eval_bioner.zip results) — same prompt construction logic, not a
reimplementation that could subtly diverge.

Usage:
    python3 build_eval_subset.py --n-per-dataset 500 --repo-id yourname/bioner-eval-subset
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import datasets
import numpy as np
import sacremoses
from tqdm.auto import tqdm

RANDOM_SEED = 42
VALIDATION_SIZE = 0.02
TEST_SIZE = 0.05
MAX_ENTITIES_PER_SAMPLE = 6

BIONER_DATASETS = [
    ("disi-unibo-nlp/ncbi", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/bc5cdr", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/bc2gm", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/bc4chemd", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/AnatEM", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/JNLPBA", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/JNLPBA-Rare", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/MedMentions-Rare", "auto", "tokens", "ner_tags"),
    ("disi-unibo-nlp/biored", "auto", "tokens", "ner_tags"),
]

SYSTEM_PROMPT = (
    "You are a precise biomedical Named Entity Recognition (NER) assistant.\n\n"
    "You will receive:\n"
    "- a TEXT (a sentence or short passage from biomedical literature);\n"
    "- a list of ENTITY TYPES (uppercase labels, e.g. \"DRUG\", \"PROCESS\").\n\n"
    "Your task: for every entity type in the list, find ALL spans in the TEXT that belong to that type.\n\n"
    "Output format — STRICT:\n"
    "1. Reply with a SINGLE JSON object. No markdown, no code fences, no commentary, no explanation.\n"
    "2. Keys MUST be exactly the entity type strings provided, with identical casing. Include every entity type, even if no span is present.\n"
    "3. Values MUST be lists of strings. Each string MUST be a VERBATIM substring of the TEXT (exact characters, exact casing). Use an empty list [] when the type is absent.\n"
    "4. Do NOT introduce extra keys (no \"start\", \"end\", \"instances\", \"entity\", \"label\", etc.).\n"
    "5. Do NOT output anything outside the JSON object — not even whitespace.\n\n"
    "Worked example:\n"
    "TEXT: \"The patient took aspirin and ibuprofen for headache.\"\n"
    "ENTITY TYPES: [\"DRUG\", \"SYMPTOM\", \"ORGANISM\"]\n"
    "CORRECT OUTPUT: {\"DRUG\": [\"aspirin\", \"ibuprofen\"], \"SYMPTOM\": [\"headache\"], \"ORGANISM\": []}\n"
)

USER_TEMPLATE = (
    "Extract the entities below from the TEXT. Reply with the JSON object only, "
    "following the format defined in the system message.\n\n"
    "TEXT:\n{text}\n\nENTITY TYPES:\n{entities}\n"
)


def _safe_literal_eval(val):
    return ast.literal_eval(val) if isinstance(val, str) else val


class IOBSpanExtractor:
    @staticmethod
    def extract(tokens, tags):
        spans = defaultdict(list)
        current_tokens, current_entity = [], None

        def flush():
            nonlocal current_tokens, current_entity
            if current_tokens and current_entity is not None:
                spans[current_entity].append(" ".join(current_tokens))
            current_tokens[:] = []
            current_entity = None

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                flush()
                current_tokens = [token]
                current_entity = tag[2:]
            elif (tag.startswith("I-") and current_entity is not None
                  and tag[2:] == current_entity):
                current_tokens.append(token)
            else:
                flush()
        flush()
        return dict(spans)


@dataclass(frozen=True)
class EntityDistribution:
    entities: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_iob_dataset(cls, train_ds, tag_field="ner_tags"):
        counter = Counter()
        for sample in tqdm(train_ds, desc="Computing entity frequencies", leave=False):
            tags = _safe_literal_eval(sample[tag_field])
            counter.update(tag[2:] for tag in tags if tag.startswith("B-"))
        if not counter:
            raise ValueError("No B- tags found.")
        entities = np.array(list(counter.keys()), dtype=object)
        freqs = np.array(list(counter.values()), dtype=np.float64)
        inv = 1.0 / freqs
        return cls(entities=entities, weights=inv / inv.sum())

    def sample(self, n, rng, exclude):
        if n <= 0:
            return []
        sampled, seen = [], set(exclude)
        for _ in range(10):
            if len(sampled) >= n:
                break
            for c in rng.choice(self.entities, size=(n - len(sampled)) * 2,
                                 p=self.weights, replace=True):
                if c not in seen:
                    sampled.append(c)
                    seen.add(c)
                    if len(sampled) == n:
                        return sampled
        return sampled


@dataclass(frozen=True)
class SFTConfig:
    max_entities: int = 10
    random_seed: int = 42
    token_field: str = "tokens"
    tag_field: str = "ner_tags"
    columns_to_remove: tuple = ("tokens", "ner_tags")
    system_prompt: str = SYSTEM_PROMPT
    user_template: str = USER_TEMPLATE


class SFTDataProcessor:
    def __init__(self, detokenizer, config=None):
        self._detokenizer = detokenizer
        self._config = config or SFTConfig()
        self._extractor = IOBSpanExtractor()
        self._distribution = None

    def fit(self, train_ds):
        self._distribution = EntityDistribution.from_iob_dataset(
            train_ds, tag_field=self._config.tag_field)
        return self

    def transform(self, ds, num_proc=None):
        if self._distribution is None:
            raise RuntimeError("Call fit() first.")
        return ds.map(
            self._convert_sample,
            with_indices=True,
            num_proc=num_proc,
            remove_columns=list(self._config.columns_to_remove),
        )

    def fit_transform(self, ds, num_proc=None):
        return self.fit(ds["train"]).transform(ds, num_proc)

    def _convert_sample(self, sample, idx):
        rng = np.random.default_rng(self._config.random_seed + idx)
        tokens = _safe_literal_eval(sample[self._config.token_field])
        tags = _safe_literal_eval(sample[self._config.tag_field])
        spans = self._extractor.extract(tokens, tags)
        positives = list(spans.keys())

        max_n = self._config.max_entities
        n_pos = min(len(positives), rng.integers(0, max_n + 1))
        chosen_pos = list(rng.choice(positives, size=n_pos, replace=False)) if positives else []
        n_neg = max_n - n_pos
        negatives = self._distribution.sample(n_neg, rng, exclude=set(positives))

        entity_list = chosen_pos + negatives
        rng.shuffle(entity_list)

        gt = {e: spans.get(e, []) for e in entity_list}
        text = self._detokenizer.detokenize(tokens)
        user_content = self._config.user_template.format(
            text=text,
            entities=json.dumps(entity_list, ensure_ascii=False),
        )
        return {
            # 2026-09-24: "all_gold" — every annotated span of the row,
            # all types, as a JSON string {TYPE: [span, ...]}. "messages"
            # only holds the gold for the types sampled into this row's
            # prompt (possibly none), which is the right gold for the
            # closed-schema models but not for the open-schema CoT model,
            # which extracts every entity it finds (see
            # src/metrics.select_gold). Computed before any rng draw, so
            # adding it changes nothing else in the row.
            "all_gold": json.dumps(spans, ensure_ascii=False),
            # 2026-09-21: "raw_text" added alongside the existing
            # "messages" field — NOT a replacement. "messages" is left
            # completely untouched (baseline/pretrain must keep using
            # the exact closed-schema prompt they always have). This
            # extra field exists so the CoT model (trained on an
            # open-schema prompt, not this closed-schema one) can be
            # evaluated with ITS OWN correct prompt, built from this
            # same underlying text, instead of being forced through the
            # closed-schema prompt it was never trained on.
            "raw_text": text,
            "messages": [
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)},
            ]
        }


def load_and_prepare(dataset_id, split_mode, token_field, tag_field,
                      validation_size=VALIDATION_SIZE, test_size=TEST_SIZE,
                      seed=RANDOM_SEED, max_entities=MAX_ENTITIES_PER_SAMPLE):
    """Verbatim from bioner-evaluation-fix.ipynb — same splitting logic,
    same SFT prompt construction, so the resulting subset matches exactly
    what a live run of that notebook would produce for the same indices."""
    raw = datasets.load_dataset(dataset_id)
    print(f"  Splits disponibili: {list(raw.keys())}")

    if split_mode == "split":
        tt = raw["train"].train_test_split(test_size=test_size, seed=seed)
        tv = tt["train"].train_test_split(
            test_size=validation_size / (1 - test_size), seed=seed)
        raw = datasets.DatasetDict({
            "train": tv["train"],
            "validation": tv["test"],
            "test": tt["test"],
        })
    else:
        if "validation" not in raw and "dev" in raw:
            raw = datasets.DatasetDict({
                "train": raw["train"],
                "validation": raw["dev"],
                "test": raw["test"],
            })

    if "train" not in raw:
        fit_split = list(raw.keys())[0]
        raw = datasets.DatasetDict({"train": raw[fit_split], **raw})

    detok = sacremoses.MosesDetokenizer(lang="en")
    cfg = SFTConfig(
        max_entities=max_entities,
        random_seed=seed,
        token_field=token_field,
        tag_field=tag_field,
        columns_to_remove=(token_field, tag_field),
    )
    proc = SFTDataProcessor(detokenizer=detok, config=cfg)
    return proc.fit_transform(raw)


def matches_existing(repo_id, ds_name, subset):
    """True if the Hub has no version of this dataset yet, or if its rows
    have exactly the same prompts (messages) as the rebuilt subset — so
    re-running this script to add a column never changes what the models
    are evaluated on."""
    try:
        existing = datasets.load_dataset(repo_id, data_files={"test": f"{ds_name}/test-*"})["test"]
    except Exception as error:
        print(f"  No existing version of {ds_name} found on the Hub ({type(error).__name__}), nothing to compare.")
        return True
    same = len(existing) == len(subset) and existing["messages"] == subset["messages"]
    print(f"  Existing {ds_name} on the Hub: {len(existing)} rows, prompts "
          f"{'identical' if same else 'DIFFERENT'} to the rebuilt ones.")
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-dataset", type=int, default=500,
                         help="Rows to sample from each dataset's test split (capped at the split's own size if smaller)")
    parser.add_argument("--repo-id", type=str, required=True, help="e.g. yourname/bioner-eval-subset")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--save-dir", type=str, default=None,
                         help="Also save each dataset's subset locally to <save-dir>/<dataset> "
                              "(datasets.save_to_disk) — usable with rescore.py --subset-dir")
    parser.add_argument("--no-push", action="store_true",
                         help="Build (and check/save) only, do not push anything to the Hub")
    parser.add_argument("--force", action="store_true",
                         help="Push even if the rebuilt rows' prompts differ from the version already on the Hub")
    args = parser.parse_args()

    if not args.no_push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        print(f"Repo {args.repo_id!r} ready (private).")

    for ds_id, split_mode, tok_field, tag_field in BIONER_DATASETS:
        ds_name = ds_id.split("/")[-1]
        print(f"\n{'=' * 60}\n{ds_name}\n{'=' * 60}")

        sft_ds = load_and_prepare(ds_id, split_mode, tok_field, tag_field, seed=args.seed)
        test_split = sft_ds["test"]
        n = min(args.n_per_dataset, len(test_split))

        # Fixed random selection of row indices, independent of eval-time
        # batching or model — this IS the "frozen" subset every future
        # evaluation reuses.
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(test_split), size=n, replace=False).tolist())
        subset = test_split.select(indices)

        print(f"  {len(test_split)} test rows available, sampled {n} (fixed indices, seed={args.seed})")

        if not args.force and not matches_existing(args.repo_id, ds_name, subset):
            raise SystemExit(
                f"  {ds_name}: the rebuilt prompts differ from the ones already on {args.repo_id} — "
                f"pushing would silently change the evaluation set that existing results were computed "
                f"on. Nothing was pushed for this dataset. Pass --force only if that is intended.")

        if args.save_dir:
            subset.save_to_disk(f"{args.save_dir}/{ds_name}")
            print(f"  Saved to {args.save_dir}/{ds_name}")
        if not args.no_push:
            subset.push_to_hub(args.repo_id, config_name=ds_name, private=True)
            print(f"  Pushed to {args.repo_id} as config '{ds_name}'")

    print("\nDone." + ("" if args.no_push else f" All 9 subsets pushed to https://huggingface.co/datasets/{args.repo_id}"))


if __name__ == "__main__":
    main()
