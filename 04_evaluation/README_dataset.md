---
license: other
task_categories:
- token-classification
language:
- en
tags:
- biomedical
- ner
- evaluation
pretty_name: "BioNER Fixed Evaluation Subset"
---

# BioNER Fixed Evaluation Subset

A **fixed, pre-built evaluation subset** for 9 standard biomedical Named
Entity Recognition (NER) benchmarks, created for reproducible pre/post
fine-tuning comparisons of MedGemma-4B (`google/medgemma-1.5-4b-it`) as
part of an internship project at DISI, University of Bologna.

## Why this exists

Comparing a model before and after fine-tuning — and later against a
further CoT-distilled version — is only meaningful if every evaluation
uses **exactly the same examples, in exactly the same prompt form**. The
underlying entity-distractor sampling used when converting IOB spans into
prompts is itself randomized (see
[disi-unibo-nlp's BioNER Datasets collection](https://huggingface.co/collections/disi-unibo-nlp/bioner-datasets)
and the SFT data pipeline this project builds on), so re-running the
conversion script twice on the "same" rows can silently produce two
different prompts. This dataset freezes that randomness once, so every
future evaluation — today, and months from now — compares models on
identical inputs.

## What's in it

Nine configs, one per source dataset, each already converted into a
ready-to-use chat-format prompt:

| Config | Source | Rows |
|---|---|---|
| `ncbi` | [disi-unibo-nlp/ncbi](https://huggingface.co/datasets/disi-unibo-nlp/ncbi) | 500 |
| `bc5cdr` | [disi-unibo-nlp/bc5cdr](https://huggingface.co/datasets/disi-unibo-nlp/bc5cdr) | 500 |
| `bc2gm` | [disi-unibo-nlp/bc2gm](https://huggingface.co/datasets/disi-unibo-nlp/bc2gm) | 500 |
| `bc4chemd` | [disi-unibo-nlp/bc4chemd](https://huggingface.co/datasets/disi-unibo-nlp/bc4chemd) | 500 |
| `AnatEM` | [disi-unibo-nlp/AnatEM](https://huggingface.co/datasets/disi-unibo-nlp/AnatEM) | 500 |
| `JNLPBA` | [disi-unibo-nlp/JNLPBA](https://huggingface.co/datasets/disi-unibo-nlp/JNLPBA) | 500 |
| `JNLPBA-Rare` | [disi-unibo-nlp/JNLPBA-Rare](https://huggingface.co/datasets/disi-unibo-nlp/JNLPBA-Rare) | 465 (full test set — smaller than 500) |
| `MedMentions-Rare` | [disi-unibo-nlp/MedMentions-Rare](https://huggingface.co/datasets/disi-unibo-nlp/MedMentions-Rare) | 500 |
| `biored` | [disi-unibo-nlp/biored](https://huggingface.co/datasets/disi-unibo-nlp/biored) | 500 |

Each row is a single split (`test`) with one column, `messages`: a
3-turn chat (`system`, `user`, `assistant`) in the exact format expected
by MedGemma's chat template. The `assistant` turn holds the gold-standard
answer (JSON object mapping entity types to verbatim text spans) used as
ground truth during evaluation — **not** meant to be shown to the model
at inference time (drop the last turn before generating).

## How the subset was built

- 500 rows sampled per dataset (or the full test set, if smaller — see
  `JNLPBA-Rare` above), using a fixed random seed (`42`) over row indices.
- For datasets without pre-existing validation/test splits, an 95/5
  train/test split was applied (same seed), matching the split logic used
  during the model's own fine-tuning.
- Prompts were built with the same `SFTDataProcessor` / `IOBSpanExtractor`
  / entity-distractor-sampling logic used to prepare the SFT training
  data — reused verbatim, not reimplemented, to avoid any drift between
  training-time and evaluation-time prompt construction.

## Licensing note

This is a derived subset of the datasets listed above, each carrying its
own original license and citation requirements (see each source repo's
own dataset card). No license is asserted over the source data itself
here — check the original repos before reuse for anything beyond
reproducing this project's own evaluation.
