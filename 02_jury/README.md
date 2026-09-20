# Jury

An LLM-jury evaluation pipeline that scores entity-extraction results
produced by another model (the "extractor", Step 1 of the wider project)
against a set of gold entity spans. Three independent judge LLMs each
score every row on two axes — **completeness** and **correctness** — and
their scores are aggregated into a single verdict per row, with a flag
for rows where the judges disagree too much to trust blindly.

It is built to run unattended for hours on a SLURM cluster, against
GPU-hosted judge models served locally via [vLLM](https://github.com/vllm-project/vllm),
with resilience against node loss, server crashes, and job time limits.

## How it works

1. **Input**: a JSONL file produced by the extractor pipeline (e.g.
   `outputs/free_cot_merged.jsonl`), one row per span, shaped like:
   ```json
   {
     "id": "ner_1",
     "span": "...text that was extracted from...",
     "gold": {"PERSON": ["Alice"], "ORG": ["Acme Corp"]},
     "reasoning": "...the extractor's chain-of-thought...",
     "entities": [{"text": "Alice", "type": "PERSON", "reason": "..."}],
     "excluded": [{"text": "...", "reason": "..."}]
   }
   ```
2. **Judging**: `main_judge.py` sends each row, rendered via the prompt
   template in `src/judge_prompts.py`, to one judge model (one process =
   one judge). The judge is asked to return only a JSON object:
   ```json
   {
     "completeness_score": 0.0,
     "completeness_notes": "...",
     "correctness_score": 0.0,
     "correctness_notes": "..."
   }
   ```
   - **Completeness**: what fraction of the gold spans are meaningfully
     covered by the extraction (by meaning, not exact string match).
   - **Correctness**: of the entities the extraction found that are
     *not* in the gold list, what fraction are genuinely justified by
     the text (rather than fabricated).

   The two scores are judged independently by design — an extraction
   with an unjustified extra entity can still be perfectly complete.
3. **Three judges, one input**: the same input file is scored three
   times, once per judge model, each writing its own output file
   (`outputs/jury_scores_<judge>.jsonl`). The judges are deliberately
   *not* from the same model family as the extractor, to avoid
   same-family bias:
   - `google/gemma-3-4b-it`
   - `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
   - `ibm-granite/granite-4.1-8b`
4. **Aggregation**: `aggregate_jury_scores.py` merges the three judges'
   outputs by row id into `outputs/jury_scores_aggregated.jsonl`, with
   mean/stdev per axis and a `flagged` boolean when judges disagree
   past a threshold (default stdev > 0.3).

## Project layout

```
main_judge.py                  Entry point: one judge model scores one input file
aggregate_jury_scores.py       Merges per-judge outputs into one verdict per row
try_judge_combinations.py      Tries every judge-subset combination, reports how many rows would pass the quality threshold under each — used to diagnose whether one judge (or a specific pairing) is disproportionately responsible for a low pass rate
src/
  config.py                    Config dataclass — all run parameters, per-judge notes
  judge_prompts.py             The jury prompt template + gold-span flattening
  judge_parser.py              Parses a judge's raw text response into a JudgeScore
  llm_client.py                Async OpenAI-compatible client (talks to vLLM), retries/backoff
  circuit_breaker.py            Trips after too many consecutive failures (dead server)
  storage.py                   Append-only JSONL writer, resumable via already-processed ids
  hf_sync.py                   Periodic backup/recovery of output files to a HF dataset repo
subset_tools/
  build_judge_subsets.py       Splits a judge's remaining (unscored) ids into N subset files
  merge_judge_parts.py         Merges per-subset judge outputs back into one file
build/
  Dockerfile                   CUDA + vLLM image the judge runs in
  requirements.txt             Pinned deps (vllm==0.8.5, torch==2.6.0, transformers==4.52.4)
train_judge.sh                 Starts a local vLLM server for one judge, runs main_judge.py, tears it down
run_docker_judge.sh            Wraps train_judge.sh in `docker run` with the right GPU/cache mounts
sbatch_launch_judge.sh         Submits one SLURM job per judge (3 jobs total) against one input file
sbatch_launch_judge_subsets.sh Per-judge sharding: builds subsets + submits N jobs per judge
outputs/subsets/               Example pre-built id subset files (subset_1.txt .. subset_4.txt)
```

## Quick start

### 1. Build the Docker image

```bash
docker build -t pilener-jury-image -f build/Dockerfile .
```

### 2. Set up secrets

Create a `.env` file in the project root (read by both `train_judge.sh`
and `main_judge.py` via `python-dotenv`):

```bash
HF_TOKEN=hf_xxx   # only needed if using --hf-backup-repo-id
```

### 3. Run one judge locally (no SLURM)

```bash
./train_judge.sh google/gemma-3-4b-it outputs/free_cot_merged.jsonl 10
```

This starts a local vLLM server for the given model, waits for it to
become healthy, scores the first 10 rows (drop the `10` for a full run,
or pass `""` then a subset file path — see the script's header comment
for exact argument order), and shuts the server down afterwards.

Sampling parameters differ per judge model (per each one's own model
card) and are set via environment variables before calling the script:

| Judge | Env vars |
|---|---|
| `google/gemma-3-4b-it` | `JUDGE_TEMPERATURE=1.0 JUDGE_TOP_P=0.95 JUDGE_TOP_K=64` |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `JUDGE_TEMPERATURE=0.6 JUDGE_TOP_P=0.95 JUDGE_MAX_TOKENS=8192 VLLM_MAX_MODEL_LEN=12288` |
| `ibm-granite/granite-4.1-8b` | `JUDGE_TEMPERATURE=1.0 JUDGE_TOP_P=0.95` |

### 4. Run against a live vLLM server directly

If you already have a vLLM (or any OpenAI-compatible) server running,
call `main_judge.py` directly instead of going through `train_judge.sh`:

```bash
python3 main_judge.py \
    --judge-model google/gemma-3-4b-it \
    --input-path outputs/free_cot_merged.jsonl \
    --base-url http://localhost:8000/v1 \
    --temperature 1.0 --top-p 0.95 --top-k 64 \
    --concurrency 8 \
    --hf-backup-repo-id yourname/jury-scores
```

Key flags (see `python3 main_judge.py --help` for the full list):
- `--id-subset-file`: only score ids listed in a subset file (see below).
- `--limit`: score only the first N rows — for smoke-testing.
- `--hf-backup-repo-id`: push the output file to a private HF dataset
  repo periodically (`--hf-push-every`, default 200 results) and on
  SIGTERM/exit, so progress survives node loss. Omit to stay local-only.
- `--circuit-breaker-threshold`: consecutive failures before the run
  stops trying new requests (default 15) — protects against hammering
  a dead vLLM server with the full retry cycle on every row.

Output goes to `outputs/jury_scores_<judge-model-basename>.jsonl` (or
`..._<judge>_<subset-name>.jsonl` if `--id-subset-file` was used).

**Re-running is always safe**: on startup, ids already present in that
judge's own output file are skipped, so a killed/interrupted job can
simply be resubmitted with the same command.

### 5. Full cluster run: all three judges in parallel (SLURM)

```bash
./sbatch_launch_judge.sh outputs/free_cot_merged.jsonl
```

Submits one `sbatch` job per judge (3 total), each running
`run_docker_judge.sh <judge> <input_path>` on its own GPU, all scoring
the same input file independently. Monitor with `squeue`. If a job dies
or hits its time limit, just resubmit the same judge's job — it resumes
automatically.

### 6. Sharding one judge across multiple parallel jobs

For faster throughput (or to keep under a per-job time limit), a single
judge's remaining work can be split into N subsets and run as N parallel
jobs:

```bash
python3 subset_tools/build_judge_subsets.py \
    outputs/free_cot_merged.jsonl \
    outputs/jury_scores_gemma-3-4b-it*.jsonl \
    --n-subsets 4 --output-dir outputs/subsets_gemma-3-4b-it
```

This computes `all_ids − already_done_ids` (from any existing output
files you pass in) and writes `subset_1.txt` .. `subset_4.txt`, one id
per line. Then launch one job per subset with
`--id-subset-file outputs/subsets_gemma-3-4b-it/subset_N.txt`.

`sbatch_launch_judge_subsets.sh` automates this for all three judges at
once (`N_SUBSETS` and `NODE` are configurable via env vars):

```bash
N_SUBSETS=4 NODE=moro232 INPUT_PATH=outputs/free_cot_merged.jsonl \
    ./sbatch_launch_judge_subsets.sh
```

Once all subset jobs for one judge finish, merge its parts back into a
single file:

```bash
python3 subset_tools/merge_judge_parts.py \
    "outputs/jury_scores_gemma-3-4b-it_subset_*.jsonl" \
    --output outputs/jury_scores_gemma-3-4b-it.jsonl
```

### 7. Aggregate all three judges into a final verdict

Once every judge has a complete (or merged) output file:

```bash
python3 aggregate_jury_scores.py \
    outputs/jury_scores_gemma-3-4b-it.jsonl \
    outputs/jury_scores_DeepSeek-R1-Distill-Llama-8B.jsonl \
    outputs/jury_scores_granite-4.1-8b.jsonl \
    --exclude-completeness-from granite-4.1-8b
```

Writes `outputs/jury_scores_aggregated.jsonl`, one row per id:

```json
{
  "id": "ner_1",
  "n_judges_completeness": 2,
  "n_judges_correctness": 3,
  "mean_completeness": 0.83,
  "mean_correctness": 1.0,
  "stdev_completeness": 0.24,
  "stdev_correctness": 0.0,
  "flagged": false
}
```

`--exclude-completeness-from` drops a named judge's `completeness_score`
from the mean (its `correctness_score` is still included normally). This
exists specifically for `granite-4.1-8b`, whose `completeness_score`
came back as exactly `0.00` across every tested row in two separate real
test rounds — zero variance across genuinely different inputs carries no
signal worth averaging in. It is not a general-purpose "distrust this
judge" switch; only use it for an axis with a confirmed, reproducible
lack of variance.

### 8. Diagnosing a low pass rate across every judge combination

If the quality-filtered dataset ends up much smaller than expected, it's
worth checking whether one judge (or one specific pairing) is
disproportionately responsible, before concluding the extractor's output
itself is at fault:

```bash
python3 try_judge_combinations.py \
    outputs/jury_scores_gemma-3-4b-it.jsonl \
    outputs/jury_scores_granite-4.1-8b.jsonl \
    outputs/jury_scores_DeepSeek-R1-Distill-Llama-8B.jsonl \
    --min-completeness 0.6 --min-correctness 0.6
```

Tries every non-empty combination of the three judges (each alone, every
pair, all three together) and reports how many rows would pass the
quality threshold under each — useful for telling apart "one judge is
systematically too harsh" from "the judges disagree on which rows are
good" (the latter shows up as a pairwise combination scoring *worse*
than either judge alone, since the two rarely agree on the same rows).

## Resilience design

The same three-layer design is used here as in the (separate) extractor
pipeline this project builds on:

- **Local JSONL is the source of truth.** Every scored row is appended
  immediately (`src/storage.py`), so a crash loses at most one in-flight
  record. Re-running a job skips ids already present in its output.
- **Optional Hugging Face backup** (`src/hf_sync.py`). If
  `--hf-backup-repo-id` is set, the output file is pushed to a private
  HF dataset repo every `--hf-push-every` results, on SIGTERM, and at
  the end of the run — and pulled back down automatically if a node
  starts with no local output file. A failed backup never crashes the
  run itself.
- **Circuit breaker** (`src/circuit_breaker.py`). After
  `--circuit-breaker-threshold` consecutive failures (default 15), the
  run stops issuing new requests instead of burning the full
  retry/backoff cycle against a vLLM server that has likely died.

## Known issues

- **Gemma-3-4B-it + vLLM**: a confirmed, isolated bug where vLLM's
  OpenAI-compatible endpoint returns empty content for this model
  specifically (the same prompts work fine via plain `transformers`).
  Splitting into subsets does not fix it. `src/llm_client.py` already
  falls back to `reasoning_content` when `content` is empty, and logs
  the full choice object if both are empty, to help diagnose it further.
- Each judge's sampling parameters (temperature/top_p/top_k/max_tokens)
  come from that model's own official card and genuinely differ — do
  not assume one hardcoded set works for all three (see `src/config.py`
  and `train_judge.sh` for the specific values and reasoning).
- **Environment variables and Docker**: a container does not
  automatically inherit the host shell's environment variables, only
  whatever is explicitly passed via `-e`. `run_docker_judge.sh` forwards
  every judge-specific variable (`JUDGE_TEMPERATURE`, `JUDGE_TOP_P`,
  `JUDGE_TOP_K`, `JUDGE_MAX_TOKENS`, `VLLM_MAX_MODEL_LEN`) explicitly —
  if you add a new configurable parameter to `train_judge.sh`, remember
  to also add it to `run_docker_judge.sh`'s `-e` flags, or it will
  silently fall back to its default with no error.

## Requirements

- Docker with NVIDIA GPU support (`nvidia/cuda:12.2.0-devel-ubuntu22.04`
  base image), or a local Python 3.11 env with `build/requirements.txt`
  installed and `torch==2.6.0+cu124` (see `build/Dockerfile`).
- A single 24GB GPU is sufficient per judge — all three models fit in
  bf16 without quantization (Gemma-3-4B ~8.6GB, DeepSeek-R1-Distill-8B /
  Granite-4.1-8B ~16GB each).
