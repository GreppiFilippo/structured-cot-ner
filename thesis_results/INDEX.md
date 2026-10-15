# Thesis results: index

Everything here is **derived** from the raw prediction files and can be rebuilt with:

```bash
python3 04_evaluation/collect_results.py --out thesis_results
```

The originals are never modified.

| File | Content |
|---|---|
| `all_results.csv` | One row per (run, dataset), 394 rows: exact typed F1 (metric v4, tutors' metric), type-agnostic IoU F1 (metric v3, used until now), P/R for both, n, parse errors, mean output length (characters), source folder and path |
| `summary_by_run.csv` | Mean over the datasets of each group (biomedical 9, CrossNER 5, Pile-NER 1) per run |
| `bootstrap_exact_vs_iou.txt` | Paired bootstrap (2,000 resamples) of the main comparisons, under both metrics (`04_evaluation/bootstrap_exact_vs_iou.py`) |

## Where the raw results are

Each run is stored as `<folder>/<dataset>/<run>/` with these files:

- `predictions.jsonl`: per sentence `idx`, `gt`, `pred_raw`, `pred_parsed`, `parse_status`, counts;
- `metrics.json`;
- `per_entity_metrics.json`;
- `worst_examples.md`.

| Folder | What | Engine |
|---|---|---|
| `modal_primary_backup/eval` | Modal account 1: Qwen3-1.7B base/v3/v4 series, Qwen3-4B base/v4/v4lo | vLLM, 6144 tok |
| `modal_primary_backup2/eval` | Modal account 1, later download: v5A/B/C (jury-threshold ablation) | vLLM |
| `modal_secondo_backup/eval` | Modal account 2: Qwen3-4B v5t, `_desc`, MedGemma (incl. `_bosfix`) | vLLM |
| `modal_terzo_backup/eval` | Modal account 3 (Oct 2026): CrossNER completion runs | vLLM |
| `04_evaluation/outputs` | Phase 1 MedGemma: `pre_sft`, `pre_sft_1536`, `post_sft` (label-only), `post_sft_cot` (open-schema CoT) | transformers |
| `04_evaluation/outputs_qwen` | Qwen3-1.7B base and v3 on the PC (listed as `*_pc_transformers`) | transformers, 4096 tok |
| `04_evaluation/outputs_modal` | Older local copy of the Modal runs; fully contained in the backups above | — |

Copies outside the repo (not used, already covered above):

- `~/Documents/outputs_eval.zip`: phase 1;
- `~/Downloads/bioner_eval`: old eval code and `all_results.json`;
- `~/Documents/uni/tirocinio/junk/bioner_eval/outputs`.

## Run names

`qwen3-1.7b_`, `qwen3-4b_` and `medgemma-1.5-4b-it_` are followed by one of these suffixes:

| Suffix | Meaning |
|---|---|
| `base` | Untouched model, thinking on |
| `base_nothink` | Thinking disabled |
| `sft_v3` | First open-vocabulary SFT |
| `sft_v4` | SFT on jury-accepted CoT (mean ≥ 0.6) |
| `sft_v4rej` | SFT on jury-rejected CoT |
| `sft_v4all` | SFT on unfiltered CoT |
| `sft_v4lo` | SFT label-only (no reasoning) |
| `sft_v4e1` | v4 trained for 1 epoch |
| `sft_v5A` / `B` / `C` | Jury ≥ 0.8 (427 examples) / random 427 / full 3,131 |
| `sft_v5t` | Teacher-target data |
| `_desc` | Zero-shot entity-type descriptions in the prompt |
| `_bosfix` | MedGemma re-evaluated after fixing the double `<bos>` |

Rows with `status = discarded (double BOS)` are the MedGemma runs superseded by `_bosfix`. The smoke tests are excluded.

## Caveats

- Exact F1 is recomputed offline from the stored `gt` and `pred_parsed`. It uses the **old query construction**: gold types plus the eval-set types, at most 6, with no distractors chosen by the tutors' protocol. It is therefore **not** the tutors' new CrossNER protocol, only the same metric.
- The gold spans of bio/CrossNER were built with `" ".join(tokens)`. Exact match therefore penalises tokenisation-spacing differences, for example `IL - 2` vs `IL-2`. This is part of why exact F1 is lower than IoU F1.
- `phase1_open_schema_cot` names its own types, so typed exact F1 is undefined for it (left empty).
- Pile-NER is a 500-row in-domain sample; there is no official test split.
- Mean output length is measured in characters. Reasoning length in tokens (the tutors' metric) will be measured on the new runs.
