# 03 — First LoRA fine-tuning of MedGemma (first phase)

This is the first MedGemma run of the thesis, on the open-schema traces
accepted by the jury; its adapter is
`structured-cot-ner/medgemma-1.5-4b-cot-distilled-lora`. All later students,
including the type-conditioned MedGemma, are trained with
`../05_openvocab_sft/train_qwen.py`.

Fine-tunes MedGemma-4B via LoRA on the jury-filtered Chain-of-Thought dataset
(produced by `../01_extraction/prepare_sft_dataset_filtered.py`) — reasoning
distillation, not label-only: the training target includes the teacher's full
reasoning trace, not just the final JSON answer.

## Setup

```bash
cp .env.example .env
nano .env   # paste HF_TOKEN, and WANDB_API_KEY if you want wandb logging

docker build -f build/Dockerfile -t cot-sft-image .
```

## Run

```bash
mkdir -p slurm_logs
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
    --output=slurm_logs/sft_%j.out \
    --error=slurm_logs/sft_%j.err \
    run_docker_sft.sh "data/train.jsonl" "yourname/MedGemma-CoT-distilled"
```

`data/train.jsonl` is the filtered dataset — download it from wherever you
pushed it (e.g. a private HF dataset repo) before launching, or point the path
at wherever it already lives.

## What `train_sft.py` does, in order

1. Loads the dataset, splits train/validation (95/5 by default)
2. **Measures real token lengths** on the actual training data with the real
   tokenizer, before picking `MAX_SEQ_LENGTH` — the earlier label-only
   experiment hardcoded `max_length=2048`, far too short for full CoT reasoning
   traces; this avoids repeating that mistake by guessing
3. Loads the base model, applies LoRA (rank 16 by default, target modules:
   `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`)
4. Trains via `trl.SFTTrainer`, with labels masked so the loss is computed only
   on the assistant's turn, not the prompt
5. Saves the adapter (not the merged model — a few hundred MB, not a full model
   copy) and pushes it to the Hugging Face repo given as the second CLI argument
6. **Round-trip verification**: reloads the just-pushed adapter from HF and
   generates one real sample, to confirm the push actually worked, not just that
   the local save succeeded

## Resuming after an interruption

Checkpoints (adapter weights + optimizer/scheduler state, enough to resume
training exactly, not just weights) are written to
`work/checkpoints/checkpoint-N/` every `--save-steps` (default: computed
automatically from dataset size, capped at 50). **Just re-run the exact same
`sbatch` command** — the script finds the latest checkpoint on its own and
resumes from there, no extra flag needed. Only the last 3 checkpoints are kept
(`save_total_limit=3`), older ones are deleted automatically to avoid filling
the disk on a long run.

The final adapter in `work/adapter/` (produced only once, at the very end of
training) is separate from the periodic checkpoints — it holds only the LoRA
weights, no optimizer state, since it's meant for inference, not for resuming
training.

## Monitoring with Weights & Biases

Set in `.env`:

```
SFT_REPORT_TO=wandb
WANDB_API_KEY=...
WANDB_PROJECT=medgemma-cot-sft   # optional, defaults to this
```

Leave `SFT_REPORT_TO` unset (or `none`) to skip wandb entirely — no other change
needed.

## Known environment quirk

The installed `trl` version's `SFTConfig` may not accept every standard
`TrainingArguments` field (a real crash was hit with `warmup_ratio` — likely
`pip`'s resolver picking an older `trl` to satisfy some other dependency's
constraint, since `trl` is deliberately left unpinned). `train_sft.py` builds
the config kwargs as a dict first and filters against what the installed
`SFTConfig` actually accepts via `inspect.signature`, printing a warning for
anything dropped instead of crashing — check the log for
`[warn] Questa versione di trl.SFTConfig
non supporta...` if a parameter you
expect to be active doesn't seem to be having an effect.

## Requirements

Same Docker/GPU requirements as `../02_jury/` (24GB GPU sufficient,
`nvidia/cuda:12.2.0-devel-ubuntu22.04` base image). `torchvision` and `Pillow`
are required even for text-only use — MedGemma is a multimodal
(image-text-to-text) model and its `AutoProcessor` always tries to load an
image-processing component.
