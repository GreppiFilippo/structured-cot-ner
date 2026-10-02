"""Builds the open-vocabulary SFT dataset for Qwen3-4B (thinking mode):
Pile-NER, restricted to the rows with a jury-filtered teacher CoT, with the
entity types to extract given in the prompt.

For every selected row:
  - system + user turn: EXACTLY the prompt used for the closed-schema
    baseline in evaluation (SYSTEM_PROMPT / USER_TEMPLATE, read from
    ../04_evaluation/build_eval_subset.py rather than copied, so the two
    cannot drift apart), with ENTITY TYPES = the row's Pile-NER gold types
    (plus --n-negative-types sampled absent types, answered with []);
  - assistant turn: "<think>\\n{teacher reasoning}\\n</think>\\n\\n{JSON}",
    where the JSON is the original Pile-NER gold {TYPE: [span, ...]}
    (spans de-duplicated per type, first-occurrence order).

Qwen3's chat template keeps the <think> block of the final assistant turn
and its generation prompt is an exact prefix of the rendered example, so
masking the loss up to the generation prompt trains on reasoning + JSON.

Every row also gets a "meta" field measuring how much the teacher's
reasoning agrees with the gold it now precedes — the reasoning was
generated for the open-schema task, so it can reject (put in "excluded")
or never mention spans the gold contains. A summary is printed at the end;
--drop-teacher-excluded-gold removes rows where the teacher explicitly
rejected a gold span.

--target teacher replaces the Pile-NER gold with the teacher's OWN
entities, produced at the end of the same reasoning that precedes them (as
in ReasoningNER, where reasoning and labels come from the same teacher call):
the requested types are the teacher's labels in the evaluation's UPPER_SNAKE
style (MedicalCondition -> MEDICAL_CONDITION), so reasoning and answer
cannot contradict each other. Rows with a teacher entity that does not occur
verbatim in the text are dropped.

Usage:
    python3 build_openvocab_dataset.py \\
        --extractions free_cot_merged.jsonl \\
        --ids-from train.jsonl \\
        --output data/openvocab_train.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

EVAL_SUBSET_BUILDER = Path(__file__).resolve().parent.parent / "04_evaluation" / "build_eval_subset.py"

_WORD_RE = re.compile(r"\w+")
# the teacher reasoning was written for the open-schema output format; a
# trace that talks about that format ("the excluded array", "entities",
# "schema", JSON) no longer matches the new target
_OLD_FORMAT_RE = re.compile(r"\bjson\b|\bexcluded (arrays?|lists?|sections?|fields?|candidates)\b|\bentities (arrays?|lists?)\b|\b(entities|excluded) (and|or) (entities|excluded)\b|\"excluded\"|\"entities\"|\bschema\b",
                            re.IGNORECASE)


# How the teacher reasoning is wrapped in the target, per student model:
#  qwen      Qwen3 thinking mode: "<think>\n...\n</think>\n\n" + answer
#  medgemma  MedGemma-1.5's own thought block, exactly as it emits it
#            zero-shot in the baseline runs: "<unused94>thought\n..." +
#            "<unused95>" + answer
REASONING_FORMATS = {
    "qwen": lambda reasoning, answer: f"<think>\n{reasoning}\n</think>\n\n{answer}",
    "medgemma": lambda reasoning, answer: f"<unused94>thought\n{reasoning}<unused95>{answer}",
    # label-only control: same examples, empty thinking block (Qwen3's own
    # format for an answer without reasoning), so the only difference from
    # "qwen" is the distilled reasoning itself
    "qwen_label_only": lambda reasoning, answer: f"<think>\n\n</think>\n\n{answer}",
}
# reasoning for the prompts that request no entity type at all (the eval
# subset has such rows, e.g. 8/100 of the first ncbi rows): the teacher never
# saw this case, and a pilot trained without it answered them with types
# copied from the system prompt's worked example
NO_TYPES_REASONING = ("No entity types are requested for this text, so there is nothing to extract "
                      "and the answer is an empty object.")


def load_eval_prompt(path: Path = EVAL_SUBSET_BUILDER) -> tuple[str, str]:
    """SYSTEM_PROMPT and USER_TEMPLATE from the evaluation subset builder,
    read with ast (no import, so none of its heavy dependencies needed)."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("SYSTEM_PROMPT", "USER_TEMPLATE"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["SYSTEM_PROMPT"], found["USER_TEMPLATE"]


def _words(s: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(s.lower()))


def _covered(span: str, candidates: list[str]) -> bool:
    """Same span, or one nested in the other, compared on word tokens."""
    w = set(_words(span))
    if not w:
        return False
    return any((c := set(_words(x))) and (w <= c or c <= w) for x in candidates)


def _mentioned(span: str, text: str) -> bool:
    w = _words(span)
    return bool(w) and f" {' '.join(w)} " in f" {' '.join(_words(text))} "


def to_upper_snake(label: str) -> str:
    """Teacher label -> the evaluation's type style: MedicalCondition ->
    MEDICAL_CONDITION, "Sports team" -> SPORTS_TEAM."""
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", label.strip())
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()


def teacher_target(row: dict) -> dict[str, list[str]] | None:
    """{TYPE: [span, ...]} from the teacher's own entities, or None when one
    of them does not occur verbatim in the text."""
    out: dict[str, list[str]] = {}
    text = row["span"].lower()
    for e in row.get("entities") or []:
        if not isinstance(e, dict) or not str(e.get("text", "")).strip():
            continue
        if str(e["text"]).lower() not in text:
            return None
        label = to_upper_snake(str(e.get("label") or e.get("type") or ""))
        if label:
            out.setdefault(label, [])
            if e["text"] not in out[label]:
                out[label].append(e["text"])
    return out


def dedupe_gold(gold: dict[str, list[str]]) -> dict[str, list[str]]:
    return {t: list(dict.fromkeys(spans)) for t, spans in gold.items()}


def agreement(row: dict, gold: dict[str, list[str]]) -> dict:
    spans = list(dict.fromkeys(s for v in gold.values() for s in v))
    ents = [e.get("text", "") for e in row.get("entities", []) if isinstance(e, dict)]
    excl = [e.get("text", "") for e in row.get("excluded", []) if isinstance(e, dict)]
    return {
        "n_gold_spans": len(spans),
        "gold_in_teacher_entities": sum(_covered(s, ents) for s in spans),
        "gold_rejected_by_teacher": sum(_covered(s, excl) and not _covered(s, ents) for s in spans),
        "gold_mentioned_in_reasoning": sum(_mentioned(s, row["reasoning"]) for s in spans),
        "teacher_entities_not_in_gold": sum(not _covered(e, spans) for e in dict.fromkeys(ents) if e),
        "reasoning_mentions_old_format": bool(_OLD_FORMAT_RE.search(row["reasoning"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractions", required=True, help="free_cot_merged.jsonl (step 1 output)")
    parser.add_argument("--ids-from", required=True,
                        help="Any .jsonl with an 'id' field listing the rows to keep, e.g. the jury-filtered "
                             "SFT set (filippogreppi/medgemma-cot-sft train.jsonl)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-negative-types", type=int, default=0,
                        help="Absent entity types added to each prompt (answered with []), sampled with "
                             "inverse frequency like the evaluation subset's distractors. Default 0: only "
                             "the row's own gold types.")
    parser.add_argument("--subsample-types", action="store_true",
                        help="Request only a random non-empty subset of each row's gold types (uniform size "
                             "1..all), answering only for those. Without it every present type is always "
                             "requested, and a fine-tuned model learns to put every entity it finds under "
                             "whatever type is requested (observed: genes returned as DISEASE on ncbi) — "
                             "the evaluation prompts likewise request only some of each text's types.")
    parser.add_argument("--keep-empty-gold", action="store_true",
                        help="Keep rows whose Pile-NER gold is empty (with 0 negatives their prompt asks "
                             "for no types at all, so they are dropped by default)")
    parser.add_argument("--drop-teacher-excluded-gold", action="store_true",
                        help="Drop rows where the teacher's reasoning explicitly rejected a gold span")
    parser.add_argument("--drop-old-format-talk", action="store_true",
                        help="Drop rows whose reasoning talks about the open-schema output format "
                             "(JSON, the excluded/entities arrays, schema)")
    parser.add_argument("--reasoning-format", choices=list(REASONING_FORMATS), default="qwen",
                        help="How the reasoning is wrapped for the student model (see REASONING_FORMATS)")
    parser.add_argument("--n-no-type-examples", type=int, default=0,
                        help="Turn this many rows with empty gold into prompts that request NO entity type "
                             "(ENTITY TYPES: []), answered with {} after a short fixed reasoning. Requires "
                             "--keep-empty-gold; the remaining empty-gold rows keep their negative types.")
    parser.add_argument("--target", choices=["gold", "teacher"], default="gold",
                        help="gold: the original Pile-NER annotation (default). teacher: the teacher's own "
                             "entities and labels, consistent with its reasoning (see module docstring)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_prompt, user_template = load_eval_prompt()
    keep_ids = {json.loads(line)["id"] for line in Path(args.ids_from).read_text().splitlines() if line.strip()}

    rows = []
    with open(args.extractions, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["id"] in keep_ids:
                    rows.append(row)
    missing = len(keep_ids) - len(rows)
    print(f"{len(rows)} selected rows found in {args.extractions}" + (f" ({missing} ids not found)" if missing else ""))

    if args.target == "teacher":
        kept = []
        for r in rows:
            t = teacher_target(r)
            if t is None:
                continue
            r["gold"] = t  # the rest of the pipeline treats it as the row's annotation
            kept.append(r)
        print(f"--target teacher: {len(rows) - len(kept)} rows dropped (teacher entity not verbatim in the text)")
        rows = kept
    type_freq = Counter(t for r in rows for t in r["gold"])
    all_types = np.array(sorted(type_freq), dtype=object)
    inv = np.array([1.0 / type_freq[t] for t in all_types])
    weights = inv / inv.sum()

    wrap = REASONING_FORMATS[args.reasoning_format]
    rng_no_types = np.random.default_rng(args.seed)
    empty_ids = [r["id"] for r in rows if not r["gold"] and reasoning_ok(r)]
    no_type_ids = set(rng_no_types.permutation(empty_ids)[: args.n_no_type_examples]) if args.keep_empty_gold else set()

    stats = Counter()
    totals = Counter()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            gold = dedupe_gold(row["gold"])
            if not reasoning_ok(row):
                stats["dropped_no_reasoning"] += 1
                continue
            if not gold and not args.keep_empty_gold:
                stats["dropped_empty_gold"] += 1
                continue
            # uses a fixed reasoning, not the teacher's, so the reasoning filters below do not apply
            if row["id"] in no_type_ids:
                user = user_template.format(text=row["span"], entities="[]")
                out.write(json.dumps({
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": wrap(NO_TYPES_REASONING, "{}")},
                    ],
                    "meta": {"kind": "no_types_requested"},
                }, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats["no_types_requested"] += 1
                continue
            meta = agreement(row, gold)
            if args.drop_teacher_excluded_gold and meta["gold_rejected_by_teacher"]:
                stats["dropped_teacher_excluded_gold"] += 1
                continue
            if args.drop_old_format_talk and meta["reasoning_mentions_old_format"]:
                stats["dropped_old_format_talk"] += 1
                continue

            types = list(gold)
            if args.subsample_types and len(types) > 1:
                rng_types = np.random.default_rng(args.seed + 1_000_003 + i)
                k = int(rng_types.integers(1, len(types) + 1))
                types = [types[j] for j in sorted(rng_types.choice(len(types), size=k, replace=False))]
                stats["rows_with_unrequested_gold_types"] += k < len(gold)
            if args.n_negative_types:
                rng = np.random.default_rng(args.seed + i)
                pool = [t for t in rng.choice(all_types, size=4 * args.n_negative_types, p=weights) if t not in gold]
                types += list(dict.fromkeys(pool))[: args.n_negative_types]
                np.random.default_rng(args.seed + i).shuffle(types)
            target = {t: gold.get(t, []) for t in types}

            user = user_template.format(text=row["span"], entities=json.dumps(types, ensure_ascii=False))
            answer = wrap(row["reasoning"].strip(), json.dumps(target, ensure_ascii=False))
            meta["reasoning_source"] = row.get("reasoning_source", "visible")
            out.write(json.dumps({
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "meta": meta,
            }, ensure_ascii=False) + "\n")
            stats["written"] += 1
            totals.update({k: int(v) for k, v in meta.items() if isinstance(v, (int, bool))})
            stats["rows_with_rejected_gold"] += bool(meta["gold_rejected_by_teacher"])
            stats["rows_fully_consistent"] += (meta["gold_in_teacher_entities"] == meta["n_gold_spans"]
                                               and meta["teacher_entities_not_in_gold"] == 0)
            stats[f"source_{meta['reasoning_source']}"] += 1

    n, g = stats["written"] - stats["no_types_requested"], max(totals["n_gold_spans"], 1)
    print(f"\n{stats['written']} examples written to {out_path} ({args.reasoning_format} format)"
          + (f", of which {stats['no_types_requested']} request no entity type" if stats["no_types_requested"] else ""))
    for k in ("dropped_no_reasoning", "dropped_empty_gold", "dropped_teacher_excluded_gold", "dropped_old_format_talk"):
        if stats[k]:
            print(f"  {stats[k]} {k.replace('_', ' ')}")
    print(f"\nTeacher reasoning vs the gold JSON it now precedes ({totals['n_gold_spans']} gold spans):")
    print(f"  gold spans the teacher also extracted:        {totals['gold_in_teacher_entities'] / g:.1%}")
    print(f"  gold spans the teacher explicitly rejected:   {totals['gold_rejected_by_teacher'] / g:.1%}"
          f"  (in {stats['rows_with_rejected_gold'] / max(n, 1):.1%} of the examples)")
    print(f"  gold spans mentioned anywhere in reasoning:   {totals['gold_mentioned_in_reasoning'] / g:.1%}")
    print(f"  teacher entities absent from the gold:        {totals['teacher_entities_not_in_gold']} "
          f"({totals['teacher_entities_not_in_gold'] / max(n, 1):.2f} per example)")
    print(f"  examples where teacher entities == gold:      {stats['rows_fully_consistent'] / max(n, 1):.1%}")
    if args.subsample_types:
        print(f"  examples where some present types are NOT requested: "
              f"{stats['rows_with_unrequested_gold_types'] / max(n, 1):.1%}")
    print(f"  reasoning talks about the old output format:  {totals['reasoning_mentions_old_format'] / max(n, 1):.1%}")
    print(f"  reasoning source: visible {stats['source_visible']}, "
          f"native_thinking_fallback {stats['source_native_thinking_fallback']}")


def reasoning_ok(row: dict) -> bool:
    return row.get("reasoning_source") != "none" and bool(row.get("reasoning", "").strip())


if __name__ == "__main__":
    main()
