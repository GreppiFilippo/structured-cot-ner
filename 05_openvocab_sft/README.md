# 05 — Open-vocabulary SFT dataset (Qwen3-4B, thinking mode)

Step 2 of the revised plan: Pile-NER restricted to the rows that have a
jury-filtered teacher CoT, in the **open-vocabulary** format used by the
evaluation benchmarks — the entity types to extract are given in the
prompt, and the target is the original Pile-NER gold.

```bash
python3 build_openvocab_dataset.py \
    --extractions free_cot_merged.jsonl \
    --ids-from train.jsonl \
    --output data/openvocab_train.jsonl
```

- `free_cot_merged.jsonl`: step 1 output (HF: `filippogreppi/free-cot-merged`)
- `train.jsonl`: any `.jsonl` with an `id` field listing the rows to keep —
  e.g. the jury-filtered set used for the MedGemma SFT
  (HF: `filippogreppi/medgemma-cot-sft`)

Each example is `{"id", "messages", "meta"}`:

- **system + user**: exactly the closed-schema prompt the baseline gets in
  evaluation (read from `../04_evaluation/build_eval_subset.py`, not
  copied), with `ENTITY TYPES` = the row's Pile-NER gold types, plus
  `--n-negative-types` absent types answered with `[]` (default 0).
- **assistant**: `<think>\n{teacher reasoning}\n</think>\n\n{JSON}` with the
  Pile-NER gold as JSON (spans de-duplicated per type — 31% of the raw
  gold spans are repeats). Checked against the `Qwen/Qwen3-4B` chat
  template: the `<think>` block of the final turn is kept, and the
  generation prompt is an exact prefix of the rendered example, so the loss
  can be masked up to it.
- **meta**: how much the teacher reasoning agrees with the gold it now
  precedes.

## Reasoning/gold agreement on the 6,895 jury-filtered rows

The teacher reasoning was generated for the open-schema task, so it does
not always lead to the gold JSON it now precedes (6,599 examples after
dropping 296 rows with empty gold):

| | |
|---|---|
| gold spans the teacher also extracted | 57.3% |
| gold spans the teacher explicitly rejected | 8.5% (in 41.5% of examples) |
| gold spans mentioned anywhere in the reasoning | 74.0% |
| examples where teacher entities == gold | 20.7% |
| reasoning that talks about the old output format (JSON, `excluded`/`entities` arrays, schema) | 30.4% (48% of the native-thinking-fallback traces) |
| reasoning source | 2,638 visible, 3,961 native-thinking fallback |

`--drop-teacher-excluded-gold` and `--drop-old-format-talk` remove the two
clearest kinds of conflict; with both, 2,877 examples remain. Whether to
filter (fewer, more consistent examples) or keep everything is an open
decision.
