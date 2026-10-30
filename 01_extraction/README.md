# 01 — Teacher extraction over Pile-NER (first phase)

The teacher (Qwen3-32B, thinking mode, served with vLLM) reasons on each
Pile-NER passage and extracts its entities without any list of types. The
merged output is published as
[`structured-cot-ner/teacher-cot-pile-ner`](https://huggingface.co/datasets/structured-cot-ner/teacher-cot-pile-ner)
(`free_cot_merged.jsonl`). The run covers all 45,889 rows: async pipeline,
resume, Hugging Face backup and circuit breaker.

**2026-08-04**: pinned to `vllm==0.8.5` after the default (`0.26.0`) turned out
to require CUDA 12.8+, above what this cluster's drivers support (12.5,
confirmed via `nvidia-smi`). Verified three independent ways: directly from
`vllm==0.8.5`'s own package metadata (requires exactly `torch==2.6.0`, the
the lab's recommended version, and `transformers>=4.51.1`); via a vLLM GitHub
issue from Qwen3's launch week confirming v0.8.4+ natively supports it; and via
[vLLM's own official recipe page for Qwen/Qwen3-32B](https://recipes.vllm.ai/Qwen/Qwen3-32B),
which states `vLLM 0.8.5+` as the model's own recommended minimum. Also checked
against a full dependency resolution (`pip install --dry-run`). **Still not yet
run on real GPU hardware** — test with a small shard first, same as every other
change in this project.

Only used once the smoke test pilot has been validated by hand — do not run this
before that.

## Setup (once — skip if you already built the image for the smoke test on

this same node)

```bash
cp .env.example .env
nano .env   # paste your real HF_TOKEN

chmod +x train.sh run_docker.sh sbatch_launch.sh

docker build -f build/Dockerfile -t pilener-cot-image .
```

## Run

```bash
./sbatch_launch.sh
squeue
```

One job, one GPU, pinned to `faretra`. Concurrency (default 24 in-flight
requests) happens inside that single vLLM instance — this is what makes a 32B
model practical on 45,889 rows in a reasonable time; without it, sequential
generation would take over a week.

## Resilience — what happens if this gets interrupted

- **Local file is the source of truth during the run.** Every successful result
  is appended immediately, one at a time.
- **Remote backup**: the accumulated output is pushed to the HF dataset repo set
  in `HF_BACKUP_REPO_ID` (top of `sbatch_launch.sh`) every 200 results, plus an
  unconditional push at the end of the run and on SIGTERM (SLURM sends this
  before killing a job that hits its time limit). Progress survives losing the
  node entirely, not just a killed process.
- **Circuit breaker**: after 15 consecutive failures (the vLLM server almost
  certainly died), the pipeline stops attempting new calls and fails the rest of
  the batch instantly instead of burning hours on retries against nothing. Check
  `vllm_server.log` if you see this in the final log line.
- **Just re-submit** (`./sbatch_launch.sh` again) after any interruption —
  `main.py` recovers the shard from the HF backup if the local file is gone, and
  always skips ids already present either way.

### Sharding across multiple parallel jobs

```bash
python3 build_subsets.py outputs/free_cot_0_45889.jsonl --n-subsets 4
```

Computes the _exact remaining id set_ (all dataset ids minus ids already present
in the output file) and writes `outputs/subset_1.txt` .. `subset_N.txt`, one id
per line. This exists instead of resuming from a contiguous index range (the old
`SPLIT_START` approach) because a contiguous range silently never retries a row
that failed earlier and fell outside a later shard's range — this script picks
up failed rows from anywhere in the dataset, not just ones past the last-seen
position.

Run each subset with:

```bash
python3 main.py --id-subset-file outputs/subset_1.txt --model-name Qwen/Qwen3-32B-AWQ ...
```

## Preparing data for SFT

```bash
python3 prepare_sft_dataset.py outputs/free_cot_0_45889.jsonl sft_ready.jsonl
```

Converts the raw output into `{"id": ..., "messages": [...]}` — the chat format
most SFT frameworks (TRL, Axolotl, LLaMA-Factory, unsloth) expect for training
the student model. Rows with no usable reasoning, or where it had to be
recovered from the model's native thinking instead of its disciplined visible
answer, are excluded by default — see the script's own docstring for the
reasoning behind that, and `--include-fallback` if you want them anyway. Safe to
run on a partial output file mid-run too, same as `check_smoke_test.py`.

## Preparing data for SFT — jury-filtered version (recommended)

```bash
python3 prepare_sft_dataset_filtered.py \
    outputs/free_cot_merged.jsonl \
    <path-to-jury-aggregated-scores>.jsonl \
    sft_ready_filtered.jsonl \
    --min-completeness 0.6 --min-correctness 0.6 \
    --keep-flagged --include-fallback
```

Same output shape as `prepare_sft_dataset.py`
(`{"id": ..., "messages":
[...]}`), but additionally filters rows by the jury's
quality verdict (see `../02_jury/`) before including them: only rows whose mean
completeness and correctness scores clear the given thresholds are kept. This is
the recommended path for building the actual SFT training set —
`prepare_sft_dataset.py` alone only filters by whether the extractor itself
produced usable reasoning, not by whether that reasoning was any good.

`--keep-flagged` keeps rows even when the three judges disagreed sharply with
each other (see the jury README's disagreement-vs-quality-threshold discussion —
disagreement alone turned out to discard most of the dataset for reasons of
judge calibration, not genuine quality). `--include-fallback` keeps rows whose
reasoning had to be recovered from the model's internal thinking block rather
than its disciplined visible answer.

## Merging (only needed if you manually split into multiple shards)

```bash
python3 merge_shards.py
```

Not needed for the default single-job, single-GPU setup above.

## What each file does

See the top-level `README.md` for full detail; in short:

- `sbatch_launch.sh` — submits the full-dataset job via `sbatch`, pinned to
  `faretra`, HF backup repo configured here
- `merge_shards.py` — combines multiple shard files into one, deduplicated by id
  (advanced/optional use only)
- `check_smoke_test.py` — same sanity-scan tool as the pilot; can be pointed at
  a partial output file mid-run too
  (`python3 check_smoke_test.py outputs/free_cot_0_45889.jsonl`)
- `main.py`, `src/` — the actual generation pipeline (async, resumable,
  circuit-breaker protected)
- `train.sh`, `run_docker.sh` — the container/vLLM launch chain invoked by
  `sbatch_launch.sh`
- `build_subsets.py` — splits remaining (not-yet-completed) ids into N subset
  files, for sharding one run across several parallel jobs
- `prepare_sft_dataset.py` — converts the raw `.jsonl` output into a chat-format
  dataset ready for SFT
- `prepare_sft_dataset_filtered.py` — same as `prepare_sft_dataset.py`, plus
  jury quality filtering (recommended over the plain version)
- `PROMPT.md` — the exact extraction prompt used by `src/prompts.py`, kept here
  as a single readable, diffable source of truth independent of the Python code
- `build/` — Docker image definition
