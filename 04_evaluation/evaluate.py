"""Evaluates ONE model (baseline, MedGemma-pretrain, or the CoT-distilled
model) on ONE of the 9 BioNER datasets, reading from the fixed subset
built by build_eval_subset.py (filippogreppi/bioner-eval-subset).

One job = one (model, dataset) pair — meant to run as parallel SLURM
jobs, each with its own dedicated GPU.

Parsing and scoring live in src/metrics.py (shared with rescore.py, which
re-scores existing predictions.jsonl files without regenerating) — see
that module's docstring for the metric definition and its version history.

Usage:
    python3 evaluate.py --model-type baseline --dataset-name ncbi
    python3 evaluate.py --model-type pretrain --dataset-name bc5cdr
    python3 evaluate.py --model-type cot --dataset-name ncbi --max-new-tokens 5120
"""
from __future__ import annotations

import argparse
import gc
import os
import re
from pathlib import Path

import datasets
import peft
import torch
import transformers
from tqdm.auto import tqdm

from src.metrics import CoTPredictionParser, PredictionParser, Scorer, select_gold, write_results
from src.prompts import FreeCoTPromptBuilder

BASE_MODEL_ID = "google/medgemma-1.5-4b-it"
PRETRAIN_PATH = "filippogreppi/MedGemma-pretrain"
COT_DISTILLED_PATH = "filippogreppi/MedGemma-CoT-distilled"
SUBSET_REPO_ID_DEFAULT = "filippogreppi/bioner-eval-subset"

_PADDING_TOKEN_RE = re.compile(r"<(?:pad|eos|bos)>")


class BatchInferencer:
    def __init__(self, model, processor, max_new_tokens=512):
        self._model, self._processor = model, processor
        self._max_new_tokens = max_new_tokens
        tok = getattr(processor, "tokenizer", processor)
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

    def disable_gradient_checkpointing(self):
        if getattr(self._model, "is_gradient_checkpointing", False):
            self._model.gradient_checkpointing_disable()

    def generate(self, prompts):
        """Returns each reply up to <end_of_turn>, with reasoning markers
        (<think>, <unused94>/<unused95>) still in place — decoded with
        skip_special_tokens=False so they survive whatever the tokenizer
        marks as special. The parsers strip them (src/metrics.clean_reply),
        and keeping them in predictions.jsonl lets rescore.py re-apply
        that step later."""
        texts = [
            self._processor.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
        ).to(self._model.device)

        try:
            with torch.inference_mode():
                outputs = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
            input_len = inputs.input_ids.shape[1]
            new_tokens = outputs[:, input_len:]
            replies = self._processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
            return [_PADDING_TOKEN_RE.sub("", r.split("<end_of_turn>")[0]).strip() for r in replies]
        finally:
            del inputs
            if "outputs" in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Model loading ---


def load_baseline():
    processor = transformers.AutoProcessor.from_pretrained(BASE_MODEL_ID, backend="torchvision")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model.eval()
    return model, processor


def _load_adapter(adapter_path, hf_token, verbose):
    processor = transformers.AutoProcessor.from_pretrained(BASE_MODEL_ID, backend="torchvision")
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model = peft.PeftModel.from_pretrained(base, adapter_path, token=hf_token)
    if verbose:
        # printed BEFORE merge_and_unload(), which discards the adapter
        # metadata — compare against the training run's LoRA settings to
        # confirm the adapter was really applied, not silently a no-op
        for adapter_name, cfg in model.peft_config.items():
            print(f"  [adapter check] name={adapter_name!r} r={cfg.r} alpha={cfg.lora_alpha} "
                  f"target_modules={sorted(cfg.target_modules)}")
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def load_pretrain(hf_token, verbose=False):
    """Label-only SFT adapter."""
    return _load_adapter(PRETRAIN_PATH, hf_token, verbose)


def load_cot_distilled(hf_token, verbose=False):
    """CoT-distilled adapter (reasoning distillation, not label-only).
    Evaluated with the same open-schema prompt it was fine-tuned on
    (src/prompts.py), not the closed-schema prompt used for the other two."""
    return _load_adapter(COT_DISTILLED_PATH, hf_token, verbose)


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Evaluation loop ---


def evaluate(model, processor, test_data, save_folder, model_type, batch_size=4, max_new_tokens=512):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    inferencer = BatchInferencer(model=model, processor=processor, max_new_tokens=max_new_tokens)
    inferencer.disable_gradient_checkpointing()

    # model_type == "cot" differs in three ways: its own open-schema
    # prompt (built from raw_text), its own output parser (a list of
    # {"text", "label"} dicts, not {TYPE: [span]}), and its own gold
    # (every annotated span, not only the types sampled into the
    # closed-schema prompt) — see src/metrics.select_gold.
    cot_prompt_builder = FreeCoTPromptBuilder() if model_type == "cot" else None
    parser = CoTPredictionParser if model_type == "cot" else PredictionParser

    scorer = Scorer()
    records = []
    gold_sources = set()
    n = len(test_data)
    for start in tqdm(range(0, n, batch_size), desc=save_folder.name, leave=False):
        batch = [test_data[i] for i in range(start, min(start + batch_size, n))]
        if model_type == "cot":
            prompts = [[{"role": "user", "content": cot_prompt_builder.build(row["raw_text"])}]
                       for row in batch]
        else:
            prompts = [row["messages"][:-1] for row in batch]

        preds_raw = inferencer.generate(prompts)

        for i, (row, pred_raw) in enumerate(zip(batch, preds_raw)):
            gold, gold_source = select_gold(row, model_type)
            gold_sources.add(gold_source)
            pred, status = parser.parse(pred_raw)
            f1, f1_ta = scorer.add(gold, pred, status)
            records.append({"idx": start + i, "f1": f1, "f1_type_agnostic": f1_ta, "parse_status": status,
                            "gt": gold, "pred_parsed": pred, "pred_raw": pred_raw})

    return write_results(save_folder, scorer, records, extra_metrics={"gold_source": sorted(gold_sources)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", choices=["baseline", "pretrain", "cot"], required=True)
    parser.add_argument("--dataset-name", required=True,
                         help="One of: ncbi, bc5cdr, bc2gm, bc4chemd, AnatEM, JNLPBA, "
                              "JNLPBA-Rare, MedMentions-Rare, biored")
    parser.add_argument("--subset-repo-id", default=SUBSET_REPO_ID_DEFAULT)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1536,
                         help="2026-08-30: raised from 512 — the baseline (non-fine-tuned) model "
                              "produces a visible reasoning preamble ('<unused94>thought...') before "
                              "its JSON answer, unlike the pretrain model which answers directly. "
                              "512 risked truncating the JSON before it appeared, on at least one "
                              "real sample observed during sanity checking. "
                              "2026-09-15: for --model-type cot, pass a much higher value explicitly "
                              "(e.g. 5120) — this model always produces a full reasoning trace before "
                              "its JSON answer. Training data itself measured up to 4077 tokens for "
                              "prompt+response combined at the 100th percentile.")
    parser.add_argument("--hf-token", default=None, help="Needed for --model-type pretrain or cot")
    parser.add_argument("--allow-missing-all-gold", action="store_true",
                         help="For --model-type cot only: run even if the subset has no 'all_gold' column, "
                              "scoring against the closed-schema prompt's gold instead. That gold omits every "
                              "entity type not sampled into the row's prompt, so the CoT model's precision "
                              "comes out artificially low — only use this for debugging.")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # "baseline"->"pre_sft" and "pretrain"->"post_sft" folder names are
    # kept as-is: aggregate scripts and resume-skip look for them by name
    subfolder = {"baseline": "pre_sft", "pretrain": "post_sft", "cot": "post_sft_cot"}[args.model_type]
    save_folder = Path(args.output_dir) / args.dataset_name / subfolder
    if (save_folder / "metrics.json").exists():
        print(f"[SKIP] {save_folder} already has metrics.json — nothing to do "
              f"(use rescore.py to recompute metrics from its predictions.jsonl).")
        return

    print(f"Loading subset '{args.dataset_name}' from {args.subset_repo_id}...")
    # explicit data_files instead of config_name: the repo's README YAML
    # config declarations broke once ("BuilderConfig 'bc5cdr' not found.
    # Available: ['default']"); the parquet layout itself is stable
    ds = datasets.load_dataset(
        args.subset_repo_id,
        data_files={"test": f"{args.dataset_name}/test-*"},
        token=hf_token,
    )
    test_data = ds["test"]
    print(f"  {len(test_data)} rows loaded.")

    if (args.model_type == "cot" and "all_gold" not in test_data.column_names
            and not args.allow_missing_all_gold):
        raise SystemExit(
            "The eval subset has no 'all_gold' column, which the CoT model needs as gold "
            "(see src/metrics.select_gold). Rebuild it first with build_eval_subset.py "
            "(prompts stay identical, checked row by row before pushing), or pass "
            "--allow-missing-all-gold to run anyway.")

    print(f"Loading {args.model_type} model...")
    if args.model_type == "baseline":
        model, processor = load_baseline()
    else:
        if not hf_token:
            raise SystemExit(f"--hf-token (or HF_TOKEN env var) is required for --model-type "
                             f"{args.model_type} (needed to pull the private LoRA adapter).")
        loader = load_pretrain if args.model_type == "pretrain" else load_cot_distilled
        model, processor = loader(hf_token, verbose=args.model_type == "cot")

    print(f"Evaluating on {args.dataset_name} ({args.model_type})...")
    metrics = evaluate(model, processor, test_data, save_folder, model_type=args.model_type,
                       batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    print(f"F1_type_agnostic={metrics['F1_micro_type_agnostic']:.4f}  "
          f"P={metrics['precision_type_agnostic']:.4f}  R={metrics['recall_type_agnostic']:.4f}  "
          f"n={metrics['n_samples']}")
    # type-exact F1 is near zero for the CoT model by construction: it
    # names its own types, never matched against the gold vocabulary
    print(f"F1_type_exact={metrics['F1_micro']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
    print(f"parse_status_counts={metrics['parse_status_counts']}  gold_source={metrics['gold_source']}")

    unload_model(model)
    print(f"Done. Results in {save_folder}")


if __name__ == "__main__":
    main()
