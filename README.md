# Structuring Chain-of-Thought for Named Entity Recognition

Code of the bachelor thesis *Structuring Universal Chain-of-thought for
Named Entity Recognition* (Department of Computer Science and Engineering,
University of Bologna, 2026). Data, models and results are on Hugging Face,
in the [`structured-cot-ner`](https://huggingface.co/structured-cot-ner)
organization.

The thesis asks whether a small language model can learn to **reason before
extracting named entities**, and whether a **structured** reasoning can be as
accurate as a free one while being shorter. A large teacher writes the
reasoning, and LoRA students with at most four billion parameters learn it.
Every query is a text plus a list of entity types, some of them absent
(distractors), and the answer is a JSON object with the mentions of each type
copied verbatim from the text.

## Two phases

**First phase.** A teacher (Qwen3-32B) reasons on Pile-NER passages without
any list of types. A jury of three small LLM judges (Gemma-3-4B-it,
DeepSeek-R1-Distill-Llama-8B, Granite-4.1-8B) verifies its outputs, and the
accepted outputs become type-conditioned training examples for Qwen3-1.7B,
Qwen3-4B and MedGemma-1.5-4B.

```
Pile-NER ─▶ 01_extraction ─▶ 02_jury ─▶ 05_openvocab_sft / 03_sft_training ─▶ 04_evaluation
            teacher CoT       3 judges    type-conditioned SFT (LoRA)            CrossNER, Pile-NER,
                                                                                 9 biomedical sets
```

**Second phase.** The teacher reasons on each query twice: freely, and in a
structured format of three one-line fields (`TYPES`, `MENTIONS`, `EDGE`).
Two students are trained on the same queries and answers, one per format
(Qwen3-4B-Thinking-2507 and SmolLM3-3B), and compared on the same sets.

## Main result (second phase, exact F1)

| Qwen3-4B-Thinking-2507 | CrossNER | Pile-NER | Biomedical | Reasoning tokens (CrossNER) |
|---|---|---|---|---|
| base, free prompt | 0.696 | 0.537 | 0.511 | 4,780 |
| student, free reasoning | **0.737** | **0.573** | **0.584** | 560 |
| student, structured reasoning | 0.712 | 0.562 | 0.563 | **386** |

Structured reasoning is learned reliably and is shorter, but it costs one to
two and a half points of F1, mostly in recall. The same pattern holds for
SmolLM3-3B.

## Repository layout

| Folder | Phase | What it does |
|---|---|---|
| [`01_extraction/`](01_extraction/README.md) | I | Teacher reasoning and open-schema extraction over Pile-NER (vLLM) |
| [`02_jury/`](02_jury/README.md) | I | Scoring of the teacher outputs by the three judges, and aggregation of their verdicts |
| [`03_sft_training/`](03_sft_training/README.md) | I | First LoRA fine-tuning of MedGemma (standalone script) |
| [`05_openvocab_sft/`](05_openvocab_sft/README.md) | I, II | Type-conditioned SFT datasets and LoRA training of all the other students (`train_qwen.py`); second-phase datasets (`build_paired_cot_dataset.py`) and launch scripts (`sbatch_paired*.sh`, `modal_app.py`) |
| [`04_evaluation/`](04_evaluation/README.md) | I, II | Evaluation sets, inference and metrics: first phase (`evaluate.py`), second phase (`eval_paired.py`, `compare_paired.py`, with the paired bootstrap), open NER baselines (`eval_baselines.py`) |

## Metric

Micro F1 with exact, case-insensitive matching of each mention within its
type, over the requested types only. Gold mentions of CrossNER and of the
biomedical sets are detokenized so that they can be copied from the text.
Differences are tested with a paired bootstrap over the queries.

## Running

Each folder's README has the setup and launch commands.

- Every stage has its own `build/Dockerfile`; images are built on the machine
  that runs them.
- The `run_docker_*.sh` scripts read secrets (such as `HF_TOKEN`) from a local
  `.env` file, see each folder's `.env.example`. Never commit a real `.env`.
- Data, model weights and predictions are not stored in this repository: each
  script reads from and writes to the Hugging Face repositories named in its
  README.

## License

[Apache 2.0](LICENSE). The MedGemma adapters are subject to the
[Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
