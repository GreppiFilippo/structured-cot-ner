# UniversalMedNER — CoT Distillation for Open-Schema Biomedical NER

Bachelor's thesis project (DISI, University of Bologna). Builds an
open-schema Named Entity Recognition pipeline that distills explicit
Chain-of-Thought reasoning — not just final labels — from a large
teacher model into a compact, deployable student (MedGemma-4B),
validated by a jury of heterogeneous judge models before being used as
training data.

## Pipeline overview

```
Pile-NER (open-schema text spans)
        |
        v
+-------------------+
| 01_extraction/     |  Teacher (Qwen3-32B) generates free-form CoT
|                     |  reasoning + open-schema entity extraction,
|                     |  no fixed type list imposed.
+-------------------+
        |
        v
+-------------------+
| 02_jury/            |  Three heterogeneous judge models (Gemma,
|                     |  DeepSeek-R1-Distill, Granite — deliberately
|                     |  different families, to avoid same-family
|                     |  bias) independently score each extraction's
|                     |  completeness and correctness.
+-------------------+
        |
        v
   (jury-filtered CoT dataset — only quality-validated rows)
        |
        v
+-------------------+
| 03_sft_training/    |  MedGemma-4B fine-tuned via LoRA on the
|                     |  filtered dataset: reasoning distillation,
|                     |  the training target includes the teacher's
|                     |  full CoT trace, not only the final answer.
+-------------------+
        |
        v
+-------------------+
| 04_evaluation/       |  The resulting model evaluated on 9 external
|                     |  biomedical NER benchmarks, against a baseline
|                     |  and a label-only-distilled model for
|                     |  comparison.
+-------------------+
```

## Repository layout

| Folder | What it does | README |
|---|---|---|
| `01_extraction/` | Teacher-generated CoT extraction over Pile-NER | [README](01_extraction/README.md) |
| `02_jury/` | Multi-judge quality scoring of the extracted reasoning | [README](02_jury/README.md) |
| `03_sft_training/` | LoRA fine-tuning of the student model | [README](03_sft_training/README.md) |
| `04_evaluation/` | Evaluation on 9 biomedical NER benchmarks | [README](04_evaluation/README.md) |

## Running order

Each stage's own README has full setup/run instructions (Docker build,
`.env` secrets, SLURM launch commands). At a high level:

1. `01_extraction/` — run the extractor, then
   `prepare_sft_dataset_filtered.py` once the jury (stage 2) has scored
   the output
2. `02_jury/` — score the extractor's output with all three judges,
   aggregate their verdicts
3. `03_sft_training/` — train the student on the filtered dataset from
   step 1 (which itself depends on step 2's scores)
4. `04_evaluation/` — evaluate the resulting adapter, compare against
   baseline and label-only-distilled results

## Infrastructure notes

- Every stage has its own `build/Dockerfile` — **images are not shared
  between cluster nodes**, rebuild on whichever node you're running on.
- Every stage's `run_docker_*.sh` reads secrets from a local `.env` file
  (see each folder's `.env.example`) — never commit a real `.env`.
- Data and model weights are **not** stored in this repository — see
  each stage's README for the relevant Hugging Face dataset/model repo.
