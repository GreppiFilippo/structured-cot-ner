# 05 — Type-conditioned SFT and the free vs structured CoT experiment

Training data and LoRA training of the students of both phases. Every
student is trained with `train_qwen.py`, except the first MedGemma run
(`../03_sft_training/`).

## Training (`train_qwen.py`)

LoRA SFT of a model in thinking format: each example is rendered with the
model's own chat template, everything up to the generation prompt is
masked, and the loss covers `<think>reasoning</think>`, the JSON answer and
the end-of-turn token. Examples longer than `--max-seq-length` are dropped,
not truncated. Defaults: LoRA r 16, alpha 32, dropout 0.05 on all attention
and MLP projections, learning rate 1e-4 with cosine schedule and 3% warmup,
2 epochs, effective batch 8, bf16, 5% validation split, best checkpoint by
validation loss, seed 42.

```bash
./run_docker_train_qwen.sh Qwen/Qwen3-4B \
    hf://structured-cot-ner/openvocab-sft/openvocab_train_v4_qwen.jsonl \
    qwen3-4b_v4 <your-hf-user>/qwen3-4b-openvocab-v4-lora
```

The same script runs on a small local GPU (`run_pilot_windows.ps1`,
`run_full_windows.ps1`) and on Modal (`modal_app.py`); only the flags change.

## First phase: type-conditioned datasets

Each example is a Pile-NER passage with a list of entity types to extract
(the gold types, optionally with absent types answered with `[]`), the teacher
reasoning in the thinking block, and a JSON answer.

| Script | Builds |
|---|---|
| `build_openvocab_dataset.py` | v3 and v4 files: teacher reasoning from the first phase (`structured-cot-ner/teacher-cot-pile-ner`) on the rows accepted by the jury (`structured-cot-ner/jury`, `jury_filtered_train.jsonl`), with the Pile-NER gold as answer |
| `modal_teacher.py` | a second teacher run (Qwen3-32B) that receives the student's prompt, with the list of types |
| `build_schema_teacher_dataset.py` | v5 files from that run: reasoning and answer from the same teacher call |

All files are in
[`structured-cot-ner/openvocab-sft`](https://huggingface.co/datasets/structured-cot-ner/openvocab-sft).
The thesis calls the versions v1, v2 and v3; the files are named v3, v4
and v5.

## Second phase: free vs structured CoT

The teacher traces come from the paired-trace dataset
`disi-unibo-nlp-students/pilener-teacher-cot-agreement-v1`: for each of
3,822 Pile-NER queries, a free trace, a structured trace (`TYPES`,
`MENTIONS`, `EDGE`) and one answer.

| Script | Does |
|---|---|
| `build_paired_cot_dataset.py` | the two training files, one per trace format, with the same queries and answers |
| `sbatch_paired.sh` | builds the data and the CrossNER and Pile-NER evaluation sets, trains the two Qwen3-4B-Thinking-2507 students, evaluates them and the base model |
| `sbatch_paired_bio.sh` | the same evaluation on the 9 biomedical sets |
| `sbatch_paired_base_structured.sh` | completes the evaluation of the base model with the structured prompt |
| `sbatch_baselines.sh` | GLiNER-L v2.1 and UniNER-7B-type on the same queries |
| `sbatch_paired_extra.sh` | follow-up runs (constrained decoding, budget forcing) not reported in the thesis |
| `modal_app.py` | the same pipeline on Modal (used for SmolLM3-3B and for the larger evaluations) |
| `run_docker_paired.sh` | runs one command inside `paired-cot-image` (`build_paired/Dockerfile`) |

The evaluation itself is `../04_evaluation/eval_paired.py`, and the
comparison with paired bootstrap is `../04_evaluation/compare_paired.py`.

Reading the paired-trace dataset needs an `HF_TOKEN` with access to it, in
a local `.env` file.
