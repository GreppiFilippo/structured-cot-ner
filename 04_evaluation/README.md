# 04 — Evaluation

Evaluates three model variants — a zero-shot baseline, a label-only
distilled model, and the CoT-distilled model this thesis builds — on 9
external biomedical NER benchmarks, using a fixed 500-row-per-dataset
evaluation subset so that all three models are compared on exactly the
same data.

## Setup

```bash
cp .env.example .env
nano .env   # paste HF_TOKEN (needed for --model-type pretrain and cot,
            # and for build_eval_subset.py's push to the Hub)

docker build -f build/Dockerfile -t bioner-eval-image .
```

## 1. Build the fixed evaluation subset (once)

```bash
python3 build_eval_subset.py --n-per-dataset 500 --repo-id filippogreppi/bioner-eval-subset
```

Samples 500 rows (deterministic, seed 42) from each of the 9 source
datasets — ncbi, bc5cdr, bc2gm, bc4chemd, AnatEM, JNLPBA, JNLPBA-Rare
(465 rows, smaller source split), MedMentions-Rare, biored — and pushes
a private dataset repo with one config per dataset. Each row carries
two things: `messages`, a closed-schema prompt (a fixed list of gold
entity types injected per row) used for the baseline and label-only
models, and `raw_text`, the underlying span text on its own, used to
build the open-schema prompt for the CoT model. Re-running with the
same seed reproduces the exact same rows — verified directly against a
backup of an earlier version of the subset, row for row identical.

## 2. Evaluate one (model, dataset) pair

```bash
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
    --output=slurm_logs/eval_baseline_ncbi_%j.out \
    --error=slurm_logs/eval_baseline_ncbi_%j.err \
    run_docker_eval.sh "baseline" "ncbi"

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
    --output=slurm_logs/eval_pretrain_ncbi_%j.out \
    --error=slurm_logs/eval_pretrain_ncbi_%j.err \
    run_docker_eval.sh "pretrain" "ncbi"

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
    --output=slurm_logs/eval_cot_ncbi_%j.out \
    --error=slurm_logs/eval_cot_ncbi_%j.err \
    run_docker_eval.sh "cot" "ncbi" --max-new-tokens 5012
```

Repeat for each of the 9 datasets and 3 model types (27 jobs total).
Each job skips itself automatically if `outputs/<dataset>/<model>/metrics.json`
already exists — safe to re-submit after an interruption. `--max-new-tokens`
should be set generously for `cot` (the model produces a full reasoning
trace before its JSON answer, not just a short answer; 5012 was used for
the reported results) — the other two model types default to 1536,
already raised from an original 512 after truncation was observed on
the baseline model's own reasoning preamble.

Output per (model, dataset) pair, under `outputs/<dataset>/<pre_sft|post_sft|post_sft_cot>/`:
- `metrics.json` — aggregate F1/precision/recall (see below for the two variants)
- `per_entity_metrics.json` — broken down by entity type
- `predictions.jsonl` — every row's gold, parsed prediction, and raw model output
- `worst_examples.md` — the 10 lowest-F1 rows, for qualitative inspection

## 3. Aggregate into the final comparison table

```bash
python3 aggregate_results_3way.py
cat comparison_table.csv
```

Writes `all_results.json` (everything found) and `comparison_table.csv`
(one row per dataset, with baseline/label-only/CoT F1 and deltas against
baseline). Safe to run before all 27 jobs finish — missing cells are
left blank rather than crashing.

## The CoT model needs its own prompt AND its own metric — two real bugs, both fixed

**Prompt**: the CoT-distilled model was fine-tuned on an open-schema
prompt (no fixed entity-type list, free-form reasoning) — built from
`raw_text` via `FreeCoTPromptBuilder` (`src/prompts.py`, a verbatim copy
of the training-time prompt). Evaluating it with the same closed-schema
prompt used for baseline/pretrain (`row["messages"]`) sends it a task
it was never trained to expect the instructions for. `evaluate.py`
branches on `--model-type` to build the right prompt for each.

**Parser**: the CoT model's real output shape is
`{"schema": [...], "entities": [{"text":..., "label":...}], "excluded": [...]}`,
not the flat `{TYPE: [span, ...]}` shape the original parser
(`PredictionParser`) expects. Feeding CoT output through
`PredictionParser` silently drops every entity (its `coerce_values`
only keeps `str`/`int`/`float` list items, and `entities` here is a
list of dicts) — confirmed directly: a real run with the wrong parser
showed apparently-empty extractions on output that, inspected raw,
contained a complete and correct reasoning trace and JSON block.
`CoTPredictionParser` handles the real shape correctly, grouping
entities by their (freely chosen) label.

**Metric**: because the CoT model is open-schema, it invents its own
type names, which rarely match the closed-schema gold vocabulary of
these benchmarks exactly (`BiologicalEntity` vs.\ the gold `DISEASE`,
for the same correctly-identified span). `evaluate.py` reports two
numbers in every `metrics.json`:
- `F1_micro` — type-exact, the original metric, comparable across all
  three models but unfairly harsh on the CoT model specifically
- `F1_micro_type_agnostic` — same IoU + Hungarian-assignment matching,
  but spans are pooled across types before matching, so a correct span
  under a different (but reasonable) type name still counts. This is
  the metric to read for the `cot` model; `aggregate_results_3way.py`
  already uses it for that column specifically, and leaves the other
  two columns on the type-exact metric.

## Known result and open question

See the top-level README's Results Summary for the full table. In
short: CoT distillation clearly corrects the severe overfitting seen in
the label-only model on some datasets (e.g.\ bc2gm, AnatEM), but does
not on average beat the zero-shot baseline across the 9 datasets. Not
yet confirmed experimentally, but plausible: the teacher's reasoning
was distilled from a general-domain corpus (Pile-NER), while MedGemma's
own zero-shot baseline already carries strong biomedical-domain priors
— the distillation may be diluting that prior rather than adding to it.

## File layout

```
build_eval_subset.py       Builds the fixed 500-row-per-dataset subset, pushes to HF
evaluate.py                 Evaluates one (model, dataset) pair
aggregate_results_3way.py   Aggregates all metrics.json into one comparison table
aggregate_results.py        Older 2-way (baseline/pretrain only) version — superseded
sanity_check.py              Quick qualitative smoke test on a handful of rows
src/prompts.py               FreeCoTPromptBuilder — verbatim copy of the training-time
                             open-schema prompt, used only for --model-type cot
run_docker_eval.sh            Wraps evaluate.py in docker run with the right GPU/cache mounts
run_docker_sanity.sh          Same, for sanity_check.py
sbatch_launch_baseline_only.sh, sbatch_launch_pretrain_only.sh, sbatch_launch_eval.sh
                             Batch launchers for subsets of the 27 (model, dataset) jobs
build/                       Docker image definition
```
