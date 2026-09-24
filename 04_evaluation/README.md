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
three things: `messages`, a closed-schema prompt (a list of entity types
per row — some real, some sampled distractors) whose assistant turn is
the gold for the baseline and label-only models; `raw_text`, the
underlying text on its own, used to build the open-schema prompt for the
CoT model; and `all_gold`, every annotated span of the row (all types,
JSON string), which is the gold for the CoT model. Re-running with the
same seed reproduces the exact same rows, and before pushing the script
checks that every rebuilt prompt is identical to the version already on
the Hub (it refuses to push otherwise, unless `--force`) — so re-running
it to add a column never changes what the models are evaluated on.

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
    run_docker_eval.sh "cot" "ncbi" --max-new-tokens 5120
```

Repeat for each of the 9 datasets and 3 model types (27 jobs total).
Each job skips itself automatically if `outputs/<dataset>/<model>/metrics.json`
already exists — safe to re-submit after an interruption. `--max-new-tokens`
should be set generously for `cot` (the model produces a full reasoning
trace before its JSON answer, not just a short answer — the reported
results used 5012) — the other two model types default to 1536,
already raised from an original 512 after truncation was observed on
the baseline model's own reasoning preamble.

Output per (model, dataset) pair, under `outputs/<dataset>/<pre_sft|post_sft|post_sft_cot>/`:
- `metrics.json` — aggregate F1/precision/recall (see below for the two variants)
- `per_entity_metrics.json` — broken down by entity type
- `predictions.jsonl` — every row's gold, parsed prediction, and raw model output
- `worst_examples.md` — the 10 lowest-F1 rows, for qualitative inspection

## 3. Re-score existing runs (CPU only)

```bash
python3 rescore.py
```

Recomputes every `metrics.json` from its `predictions.jsonl` with the
current parser and metric (`src/metrics.py`), without loading any model.
Run it whenever the metric changes (`METRIC_VERSION` in
`src/metrics.py`), so all 27 runs are scored by exactly the same code no
matter when they were generated. The previous `metrics.json` is kept once
as `metrics_before_rescore.json`. Re-scoring the CoT runs needs the
subset's `all_gold` column (rebuild the subset first, step 1).

## 4. Significance (CPU only)

```bash
python3 significance.py
```

Paired bootstrap 95% confidence intervals for every model-vs-model F1
difference, per dataset and for the 9-dataset mean, on both metric
variants (see below). Reads only the per-row counts stored in
`predictions.jsonl`; writes `outputs/significance.csv`.

## 5. Aggregate into the final comparison table

```bash
python3 aggregate_results_3way.py
cat comparison_table.csv
```

Writes `all_results.json` (everything found) and `comparison_table.csv`
(one row per dataset: type-agnostic F1/P/R for all three models, deltas
against baseline, plus type-exact F1 for baseline and label-only). Safe to
run before all 27 jobs finish — missing cells are left blank rather than
crashing. It warns about any run scored with an older metric version, or
a CoT run not scored against `all_gold` — run `rescore.py` for those.

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
entities by their (freely chosen) label, and takes the last JSON object
with an `entities` key (the answer always follows the reasoning).

**Gold**: the closed-schema gold in `messages` only covers the entity
types sampled into that row's prompt — possibly none of the real ones,
since the number of real types per prompt is drawn at random from 0 up.
That is the right gold for baseline/label-only, which are only asked
about those types. The CoT model is not told any types and extracts
everything, so against that gold every real entity of an unsampled type
would count as a false positive. It is scored against `all_gold` instead
(`src/metrics.select_gold`); `evaluate.py --model-type cot` refuses to
run on a subset without that column. Even with the full gold, the CoT
model can still extract entity kinds that a benchmark never annotates
(e.g. genes in ncbi, which only annotates diseases) — those still count
as false positives, so its recall is the easier number to interpret.

**Metric**: because the CoT model is open-schema, it invents its own
type names, which rarely match the closed-schema gold vocabulary of
these benchmarks exactly (`BiologicalEntity` vs.\ the gold `DISEASE`,
for the same correctly-identified span). `evaluate.py` reports two
numbers in every `metrics.json`:
- `F1_micro` — type-exact: IoU + Hungarian matching within each type,
  type names must match. Meaningful for baseline and label-only, near
  zero by construction for the CoT model.
- `F1_micro_type_agnostic` — same matching, but spans are pooled across
  types first, so a correct span under a different type name still
  counts. This is the only metric computed the same way for all three
  models, so `aggregate_results_3way.py` uses it for every headline
  column and every delta (earlier versions compared the CoT model's
  type-agnostic F1 against the other two models' type-exact F1).

**Requested types only** (metric version 3): for every model that is given
a list of entity types (baseline, label-only, CoT-typed), each metric is
also reported with the predicted keys that were not requested ignored
(`requested_types_only` in `metrics.json`, "req. types" columns in the
table). The prompt says the keys MUST be exactly the requested types, and
the label-only model systematically adds its own training-time types
(`DRUG`, `PROCESS`, `ORGANISM`, `SYMPTOM`, ...) to every answer — on
ncbi, bc2gm and AnatEM 85–89% of its keys were never requested — whose
spans the plain metric counts as false positives. With them ignored the
label-only model goes from 0.438 to 0.513 mean F1 and is significantly
above the baseline (+0.040, 95% CI [+0.026, +0.056]), while the baseline
barely changes (0.471 → 0.473). Report both variants.

Metric version 2 (2026-09-24, see `src/metrics.py`) changed three things
for all models alike: span tokens are word characters only, so
punctuation spacing no longer matters (gold spans are space-joined IOB
tokens like `"( CD25 )"`, while the text shown to the model is
detokenized, `"(CD25)"`); repeated spans are de-duplicated within a
type; and reasoning blocks (`<think>…</think>`, MedGemma's
`<unused94>thought…<unused95>`) are stripped before parsing, so a JSON
draft written while reasoning is never taken as the answer.

## CoT model with the requested types in its prompt (`cot_typed`)

`--model-type cot_typed` evaluates the same CoT-distilled adapter with its
training prompt kept verbatim plus one inserted section listing the
requested entity types (the same list the baseline gets) and stating that
generic biomedical concepts count as entities (`src/prompts.TYPED_SECTION`).
It is scored like the closed-schema models, against the same gold, so it
is directly comparable to them. The model was never trained with a type
list, so check schema adherence on a few rows first — `sanity_check.py`
prints how many answers used only the requested types:

```bash
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
    --output=slurm_logs/sanity_cot_typed_%j.out --error=slurm_logs/sanity_cot_typed_%j.err \
    run_docker_sanity.sh cot_typed ncbi --n-samples 10 --max-new-tokens 5120
```

Then run `run_docker_eval.sh "cot_typed" <dataset> --max-new-tokens 5120`
for the 9 datasets (results in `outputs/<dataset>/post_sft_cot_typed/`).
Here "all" measures the spans found regardless of their label, and
"requested types only" additionally requires the requested label.

## Known result and open question

The summary below predates metric version 2 and the CoT
gold fix — re-check it against `comparison_table.csv` after re-scoring.
In short: CoT distillation clearly corrects the severe overfitting seen in
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
rescore.py                  Recomputes metrics from existing predictions.jsonl (no GPU)
significance.py             Paired bootstrap CIs for model-vs-model differences (no GPU)
aggregate_results_3way.py   Aggregates all metrics.json into one comparison table
aggregate_results.py        Older 2-way (baseline/pretrain only) version — superseded
sanity_check.py              Quick qualitative smoke test on a handful of rows
src/metrics.py               Reply cleaning, parsers, gold selection, IoU/Hungarian scoring
src/prompts.py               FreeCoTPromptBuilder — verbatim copy of the training-time
                             open-schema prompt, used only for --model-type cot
run_docker_eval.sh            Wraps evaluate.py in docker run with the right GPU/cache mounts
run_docker_sanity.sh          Same, for sanity_check.py
sbatch_launch_baseline_only.sh, sbatch_launch_pretrain_only.sh, sbatch_launch_eval.sh
                             Batch launchers for subsets of the 27 (model, dataset) jobs
build/                       Docker image definition
```
