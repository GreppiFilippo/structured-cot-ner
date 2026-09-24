"""Quick sanity check: loads ONE model on a HANDFUL of rows from ONE
dataset, generates, and prints ground truth vs. prediction side by side —
so you can confirm the model loads and generates sensible JSON before
committing to a full 500-row x 9-dataset run through SLURM.

Same idea as bioner-evaluation-fix.ipynb's own "5b. Sanity check" cell —
reuses evaluate.py's actual loading/inference code directly (not a
reimplementation), so a pass here means the real evaluation will at least
get past model loading and generation.

Usage (run directly with docker run, not sbatch — this should take under
a minute, no need to queue):
    docker run --rm --gpus all -v $(pwd):/workspace -e HF_TOKEN=$HF_TOKEN \
        bioner-eval-image python3 /workspace/sanity_check.py --model-type pretrain

    docker run --rm --gpus all -v $(pwd):/workspace \
        bioner-eval-image python3 /workspace/sanity_check.py --model-type baseline
"""
from __future__ import annotations

import argparse
import json
import os

import datasets

from evaluate import (
    BASE_MODEL_ID,
    SUBSET_REPO_ID_DEFAULT,
    BatchInferencer,
    build_prompts,
    load_baseline,
    load_cot_distilled,
    load_pretrain,
    requested_types,
    unload_model,
)
from src.metrics import CoTPredictionParser, PredictionParser
from src.prompts import FreeCoTPromptBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", choices=["baseline", "pretrain", "cot", "cot_typed"], required=True)
    parser.add_argument("--dataset-name", default="ncbi", help="Which of the 9 to sanity-check against")
    parser.add_argument("--subset-repo-id", default=SUBSET_REPO_ID_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"=== SANITY CHECK: {args.model_type} on {args.dataset_name} ({args.n_samples} samples) ===\n")
    print(f"Base model: {BASE_MODEL_ID}")

    print(f"\nLoading {args.n_samples} rows from '{args.dataset_name}'...")
    ds = datasets.load_dataset(args.subset_repo_id, data_files={"test": f"{args.dataset_name}/test-*"}, token=hf_token)
    samples = [ds["test"][i] for i in range(min(args.n_samples, len(ds["test"])))]
    print(f"  Loaded {len(samples)} rows (dataset has {len(ds['test'])} total).")

    print(f"\nLoading {args.model_type} model — this is the step most likely to fail "
          "if a dependency is missing (torchvision/Pillow/accelerate) or the LoRA "
          "adapter can't be pulled (check HF_TOKEN for --model-type pretrain)...")
    if args.model_type == "baseline":
        model, processor = load_baseline()
    else:
        if not hf_token:
            raise SystemExit(f"--hf-token (or HF_TOKEN env var) is required for --model-type {args.model_type}.")
        loader = load_pretrain if args.model_type == "pretrain" else load_cot_distilled
        model, processor = loader(hf_token, verbose=True)
    print("  Model loaded OK.")

    inferencer = BatchInferencer(model=model, processor=processor, max_new_tokens=args.max_new_tokens)
    inferencer.disable_gradient_checkpointing()

    print(f"\nGenerating on {len(samples)} sample(s)...\n")
    ok_count = 0
    on_schema_count = 0
    parser_cls = CoTPredictionParser if args.model_type.startswith("cot") else PredictionParser
    cot_builder = FreeCoTPromptBuilder()
    for i, sample in enumerate(samples):
        prompt = build_prompts([sample], args.model_type, cot_builder)[0]
        gt_json = sample["messages"][-1]["content"]
        pred_raw = inferencer.generate([prompt])[0]
        pred_parsed, status = parser_cls.parse(pred_raw)
        asked = requested_types(sample)
        off_schema = sorted(k for k, v in pred_parsed.items() if v and k not in asked)
        on_schema_count += int(not off_schema)

        is_ok = status in ("ok", "fence_stripped", "extracted_object")
        ok_count += int(is_ok)

        print(f"--- sample {i} ---")
        print(f"  parse_status: {status}  {'OK' if is_ok else 'FAILED TO PARSE'}")
        print(f"  requested types: {asked}   non-requested types used: {off_schema or 'none'}")
        print(f"  GT   : {gt_json[:200]}")
        print(f"  PRED (full, {len(pred_raw)} chars): {pred_raw!r}")
        if is_ok:
            print(f"  parsed: {json.dumps(pred_parsed, ensure_ascii=False)[:200]}")
        print()

    unload_model(model)

    print("=" * 60)
    print(f"{on_schema_count}/{len(samples)} answers used only the requested entity types "
          f"(for cot_typed: the key check before launching the full evaluation).")
    if ok_count == len(samples):
        print(f"PASS: {ok_count}/{len(samples)} samples produced parseable JSON. "
              "Safe to launch the full run.")
    elif ok_count > 0:
        print(f"PARTIAL: {ok_count}/{len(samples)} samples parsed OK. "
              "Model loads and generates, but parsing isn't fully reliable — "
              "worth a closer look at the raw predictions above before committing "
              "to the full run.")
    else:
        print(f"FAIL: 0/{len(samples)} samples parsed. Model loaded and generated "
              "something, but never valid JSON — check the raw PRED text above; "
              "this usually means a prompt/chat-template mismatch, not a dependency issue.")


if __name__ == "__main__":
    main()
