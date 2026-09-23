"""Evaluates ONE model (baseline, MedGemma-pretrain, or the CoT-distilled
model) on ONE of the 9 BioNER datasets, reading from the fixed subset
built by build_eval_subset.py (filippogreppi/bioner-eval-subset).

One job = one (model, dataset) pair — meant to run as parallel SLURM
jobs, each with its own dedicated GPU.

Reuses PredictionParser / BatchInferencer / IoU-Hungarian metric verbatim
from bioner-evaluation-fix.ipynb — same evaluation code path as the
existing 100/500-sample results, so numbers stay comparable across all
three model types.

Usage:
    python3 evaluate_cot.py --model-type baseline --dataset-name ncbi
    python3 evaluate_cot.py --model-type pretrain --dataset-name bc5cdr
    python3 evaluate_cot.py --model-type cot --dataset-name ncbi --max-new-tokens 3072
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import datasets
import numpy as np
import peft
import torch
import transformers
from scipy import optimize as sp_optimize
from tqdm.auto import tqdm

from src.prompts import FreeCoTPromptBuilder

BASE_MODEL_ID = "google/medgemma-1.5-4b-it"
PRETRAIN_PATH = "filippogreppi/MedGemma-pretrain"
COT_DISTILLED_PATH = "filippogreppi/MedGemma-CoT-distilled"
SUBSET_REPO_ID_DEFAULT = "filippogreppi/bioner-eval-subset"

# --- Metric / parsing helpers (verbatim from bioner-evaluation-fix.ipynb) ---


def _tokenize(span):
    return set(span.lower().split())


def iou(a, b):
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_entity_counts(gt_spans, pred_spans):
    gc_, pc = len(gt_spans), len(pred_spans)
    if gc_ == 0 or pc == 0:
        return 0.0, gc_, pc
    M = np.array([[iou(g, p) for p in pred_spans] for g in gt_spans])
    r, c = sp_optimize.linear_sum_assignment(-M)
    return float(M[r, c].sum()), gc_, pc


def f1_from_counts(tp, gt, pred):
    d = gt + pred
    return (2.0 * tp / d) if d else 1.0


class ParseStatus:
    OK = "ok"
    FENCE_STRIPPED = "fence_stripped"
    EXTRACTED_OBJECT = "extracted_object"
    WRONG_TOP_LEVEL = "wrong_top_level"
    PARSE_ERROR = "parse_error"


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class PredictionParser:
    @classmethod
    def parse(cls, raw_text):
        text = raw_text.strip()
        for candidate, status in [
            (text, ParseStatus.OK),
            (_FENCE_RE.sub("", text).strip(), ParseStatus.FENCE_STRIPPED),
            (cls._first_json_object(text) or "", ParseStatus.EXTRACTED_OBJECT),
        ]:
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                return {}, ParseStatus.WRONG_TOP_LEVEL
            return cls.coerce_values(obj), status
        return {}, ParseStatus.PARSE_ERROR

    @staticmethod
    def _first_json_object(text):
        depth, start, in_string, escape = 0, None, False, False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
                start = start if start is not None else i
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i + 1]
        return None

    @staticmethod
    def coerce_values(obj):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, list):
                cleaned[k] = [str(x) for x in v if isinstance(x, (str, int, float))]
            elif isinstance(v, str):
                cleaned[k] = [v]
            else:
                cleaned[k] = []
        return cleaned


class CoTPredictionParser:
    """Parses the CoT model's real output shape: free-form reasoning
    prose, followed by a JSON block of {"schema": [...], "entities":
    [{"text":..., "label":..., "reason":...}], "excluded": [...]} — NOT
    the flat {TYPE: [span, span...]} shape PredictionParser expects.
    Using PredictionParser.coerce_values on this shape silently drops
    every entity (dicts aren't str/int/float, so the list comes out
    empty) — confirmed by direct test, not a hypothetical risk.

    Returns the SAME {TYPE: [span, ...]} shape PredictionParser does, by
    grouping the "entities" list by its "label" field, so the existing
    per-type compute_entity_counts loop in evaluate() works unchanged
    for the type-exact metric. Also exposes the flat span list alone
    (ignoring label) via flat_spans(), for the type-agnostic metric —
    the CoT model is open-schema and may legitimately use a different
    (but semantically equivalent) type name than the closed-schema gold
    label, which the type-exact metric would otherwise unfairly punish.
    """

    @classmethod
    def parse(cls, raw_text):
        text = raw_text.strip()
        for candidate, status in [
            (text, ParseStatus.OK),
            (_FENCE_RE.sub("", text).strip(), ParseStatus.FENCE_STRIPPED),
            (cls._first_json_object(text) or "", ParseStatus.EXTRACTED_OBJECT),
        ]:
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "entities" not in obj:
                continue
            return cls._group_by_label(obj), status
        return {}, ParseStatus.PARSE_ERROR

    @staticmethod
    def _first_json_object(text):
        # same brace-matching logic as PredictionParser._first_json_object
        depth, start, in_string, escape = 0, None, False, False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
                start = start if start is not None else i
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _group_by_label(obj):
        grouped = defaultdict(list)
        for e in obj.get("entities", []):
            if isinstance(e, dict) and isinstance(e.get("text"), str) and e.get("text"):
                label = e.get("label") if isinstance(e.get("label"), str) and e.get("label") else "UNLABELED"
                grouped[label].append(e["text"])
        return dict(grouped)

    @staticmethod
    def flat_spans(grouped: dict) -> list:
        """All spans across every type, type ignored — for the
        type-agnostic metric."""
        return [span for spans in grouped.values() for span in spans]


_ASKED_RE = re.compile(r"(?:Entities|ENTITY\s+TYPES)\s*:\s*(\[.*?\])", re.IGNORECASE | re.DOTALL)


def extract_asked_entities(user_content):
    m = _ASKED_RE.search(user_content)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(1))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


INFERENCER_DEBUG = False


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
            replies = self._processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            replies = [_THINK_RE.sub("", r).split("<end_of_turn>")[0].strip() for r in replies]
            return replies
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


def load_pretrain(hf_token, verbose=False):
    processor = transformers.AutoProcessor.from_pretrained(BASE_MODEL_ID, backend="torchvision")
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model = peft.PeftModel.from_pretrained(base, PRETRAIN_PATH, token=hf_token)
    if verbose:
        # 2026-08-30: added after being asked "how do I know the LoRA
        # adapter was actually applied, not just that loading didn't
        # error?" — merge_and_unload() below converts the PeftModel back
        # into a plain model, discarding this adapter metadata entirely,
        # so it has to be printed BEFORE that call or it's gone for good.
        # Compare these numbers against the training notebook's own
        # LORA_R / LORA_ALPHA / target_modules to confirm this is really
        # the same adapter, not a silent no-op.
        for adapter_name, cfg in model.peft_config.items():
            print(f"  [adapter check] name={adapter_name!r} r={cfg.r} alpha={cfg.lora_alpha} "
                  f"target_modules={sorted(cfg.target_modules)}")
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def load_cot_distilled(hf_token, verbose=False):
    """Loads the CoT-distilled adapter (reasoning distillation, not
    label-only). Same loading pattern as load_pretrain(), separate
    function since the adapter repo differs and this model is expected
    to generate a full reasoning trace before its JSON answer — the
    prompt/schema used during evaluation must match what it was
    fine-tuned on (open-schema CoT prompt, teacher framing kept as-is
    per an explicit decision not to rebuild the training set with a
    cleaned-up student prompt)."""
    processor = transformers.AutoProcessor.from_pretrained(BASE_MODEL_ID, backend="torchvision")
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model = peft.PeftModel.from_pretrained(base, COT_DISTILLED_PATH, token=hf_token)
    if verbose:
        for adapter_name, cfg in model.peft_config.items():
            print(f"  [adapter check] name={adapter_name!r} r={cfg.r} alpha={cfg.lora_alpha} "
                  f"target_modules={sorted(cfg.target_modules)}")
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Evaluation loop ---


def evaluate(model, processor, test_data, save_folder, model_type, batch_size=4, max_new_tokens=512, worst_k=10):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    inferencer = BatchInferencer(model=model, processor=processor, max_new_tokens=max_new_tokens)
    inferencer.disable_gradient_checkpointing()

    # 2026-09-21: model_type == "cot" branches into its own prompt AND
    # its own parser — two separate, confirmed real bugs, not one:
    # (1) reusing row["messages"][:-1] sends the CLOSED-schema prompt
    #     (built for baseline/pretrain) to a model trained on an
    #     open-schema prompt it has never seen the instructions for;
    # (2) even with the right prompt, PredictionParser.coerce_values
    #     silently drops every entity from the CoT model's real
    #     {"entities": [{"text":..., "label":...}]} shape (dicts aren't
    #     str/int/float, so they're filtered out) — confirmed by direct
    #     test, not a hypothetical. CoTPredictionParser handles that
    #     shape correctly instead.
    cot_prompt_builder = FreeCoTPromptBuilder() if model_type == "cot" else None

    n = len(test_data)
    tp_total = gt_total = pred_total = 0.0
    tp_total_ta = gt_total_ta = pred_total_ta = 0.0  # type-agnostic accumulators
    per_entity_acc = defaultdict(lambda: [0.0, 0, 0])
    parse_counts = Counter()
    records = []

    pred_path = save_folder / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as pf:
        for start in tqdm(range(0, n, batch_size), desc=save_folder.name, leave=False):
            end = min(start + batch_size, n)
            batch = [test_data[i] for i in range(start, end)]
            if model_type == "cot":
                prompts = [[{"role": "user", "content": cot_prompt_builder.build(row["raw_text"])}]
                           for row in batch]
            else:
                prompts = [row["messages"][:-1] for row in batch]
            gt_jsons = [row["messages"][-1]["content"] for row in batch]

            preds_raw = inferencer.generate(prompts)

            for i, (row, gt_json, pred_raw) in enumerate(zip(batch, gt_jsons, preds_raw)):
                try:
                    gt = PredictionParser.coerce_values(json.loads(gt_json))
                except Exception:
                    gt = {}

                if model_type == "cot":
                    pred, status = CoTPredictionParser.parse(pred_raw)
                else:
                    pred, status = PredictionParser.parse(pred_raw)
                parse_counts[status] += 1

                tp_s = gt_s = pred_s = 0.0
                for etype in set(gt) | set(pred):
                    tp, gcount, pcount = compute_entity_counts(gt.get(etype, []), pred.get(etype, []))
                    tp_s += tp
                    gt_s += gcount
                    pred_s += pcount
                    acc = per_entity_acc[etype]
                    acc[0] += tp
                    acc[1] += gcount
                    acc[2] += pcount

                tp_total += tp_s
                gt_total += gt_s
                pred_total += pred_s
                f1 = f1_from_counts(tp_s, gt_s, pred_s)

                # type-agnostic: same IoU+Hungarian matching, but on the
                # flat span lists with type ignored entirely — an
                # open-schema model using "Illness" instead of the gold
                # "DISEASE" for the same correct span should not be
                # punished as both a false negative and a false positive.
                gt_flat = [s for spans in gt.values() for s in spans]
                pred_flat = [s for spans in pred.values() for s in spans]
                tp_ta, gt_c_ta, pred_c_ta = compute_entity_counts(gt_flat, pred_flat)
                tp_total_ta += tp_ta
                gt_total_ta += gt_c_ta
                pred_total_ta += pred_c_ta

                rec = {"idx": start + i, "f1": f1, "parse_status": status,
                       "gt": gt, "pred_parsed": pred, "pred_raw": pred_raw}
                records.append(rec)
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    f1_micro = f1_from_counts(tp_total, gt_total, pred_total)
    precision = (tp_total / pred_total) if pred_total else 0.0
    recall = (tp_total / gt_total) if gt_total else 0.0
    f1_micro_ta = f1_from_counts(tp_total_ta, gt_total_ta, pred_total_ta)
    precision_ta = (tp_total_ta / pred_total_ta) if pred_total_ta else 0.0
    recall_ta = (tp_total_ta / gt_total_ta) if gt_total_ta else 0.0
    metrics = {
        "n_samples": n,
        "F1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
        "F1_micro_type_agnostic": f1_micro_ta,
        "precision_type_agnostic": precision_ta,
        "recall_type_agnostic": recall_ta,
        "TP": float(tp_total), "GT": int(gt_total), "PRED": int(pred_total),
        "parse_status_counts": dict(parse_counts),
    }
    with open(save_folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    per_ent = {}
    for etype, (tp, gcount, pcount) in per_entity_acc.items():
        if gcount == 0 and pcount == 0:
            continue
        per_ent[etype] = {
            "precision": (tp / pcount) if pcount else 0.0,
            "recall": (tp / gcount) if gcount else 0.0,
            "f1": f1_from_counts(tp, gcount, pcount),
            "gt_count": int(gcount), "pred_count": int(pcount),
        }
    per_ent = dict(sorted(per_ent.items(), key=lambda kv: kv[1]["f1"]))
    with open(save_folder / "per_entity_metrics.json", "w") as f:
        json.dump(per_ent, f, indent=2)

    worst = sorted(records, key=lambda r: r["f1"])[:worst_k]
    with open(save_folder / "worst_examples.md", "w") as f:
        for r in worst:
            f.write(f"## idx {r['idx']}  F1={r['f1']:.3f}\n")
            f.write(f"gt: {json.dumps(r['gt'], ensure_ascii=False)}\n")
            f.write(f"pred: {r['pred_raw'][:300]}\n---\n")

    return metrics


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
                              "(e.g. 3072) — this model was trained on the ORIGINAL teacher-framed "
                              "open-schema CoT prompt (kept as-is, not the cleaned-up student prompt), "
                              "so it always produces a full reasoning trace before its JSON answer, "
                              "not just a short answer. Training data itself measured up to 4077 "
                              "tokens for prompt+response combined at the 100th percentile.")
    parser.add_argument("--hf-token", default=None, help="Needed for --model-type pretrain or cot")
    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # 2026-09-15: "baseline"->"pre_sft" and "pretrain"->"post_sft" kept
    # EXACTLY as before — renaming either would break resume-skip logic
    # against the 9 already-completed real evaluation runs sitting in
    # those exact folder names, and would break aggregate_results.py,
    # which looks for "pre_sft"/"post_sft" by name. Only "cot" gets a
    # new, third folder name that doesn't collide with either.
    subfolder = {"baseline": "pre_sft", "pretrain": "post_sft", "cot": "post_sft_cot"}[args.model_type]
    save_folder = Path(args.output_dir) / args.dataset_name / subfolder
    if (save_folder / "metrics.json").exists():
        print(f"[SKIP] {save_folder} already has metrics.json — nothing to do.")
        return

    print(f"Loading subset '{args.dataset_name}' from {args.subset_repo_id}...")
    # 2026-08-30: loading via explicit data_files instead of relying on
    # config_name resolution through the repo's own README YAML — a real
    # failure ("BuilderConfig 'bc5cdr' not found. Available: ['default']")
    # showed the repo's config declarations can go stale/break (e.g. if
    # the auto-generated YAML front-matter gets disturbed while editing
    # the README), silently collapsing all 9 configs into one generic
    # 'default'. Pointing straight at the known file layout
    # (<dataset_name>/test-*.parquet, confirmed from the repo's own
    # auto-generated metadata) sidesteps that fragility entirely.
    ds = datasets.load_dataset(
        args.subset_repo_id,
        data_files={"test": f"{args.dataset_name}/test-*"},
        token=hf_token,
    )
    test_data = ds["test"]
    print(f"  {len(test_data)} rows loaded.")

    print(f"Loading {args.model_type} model...")
    if args.model_type == "baseline":
        model, processor = load_baseline()
    elif args.model_type == "pretrain":
        if not hf_token:
            raise SystemExit("--hf-token (or HF_TOKEN env var) is required for --model-type pretrain "
                              "(needed to pull the private LoRA adapter).")
        model, processor = load_pretrain(hf_token)
    else:  # cot
        if not hf_token:
            raise SystemExit("--hf-token (or HF_TOKEN env var) is required for --model-type cot "
                              "(needed to pull the private LoRA adapter).")
        model, processor = load_cot_distilled(hf_token, verbose=True)

    print(f"Evaluating on {args.dataset_name} ({args.model_type})...")
    metrics = evaluate(model, processor, test_data, save_folder, model_type=args.model_type,
                        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    print(f"F1={metrics['F1_micro']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"n={metrics['n_samples']}")
    if args.model_type == "cot":
        # 2026-09-22: printed separately for "cot" — F1_micro (type-exact)
        # is expected to stay near zero for an open-schema model that
        # invents its own type names, and printing only that number to
        # the console made a real, working run look like a total failure
        # (confirmed: parse_status_counts showed 499/500 successful
        # parses on a run whose console F1 read exactly 0.0000).
        print(f"F1_type_agnostic={metrics['F1_micro_type_agnostic']:.4f}  "
              f"P_type_agnostic={metrics['precision_type_agnostic']:.4f}  "
              f"R_type_agnostic={metrics['recall_type_agnostic']:.4f}  "
              f"(this is the metric that matters for an open-schema model — "
              f"F1_micro above compares type NAMES exactly, which this model "
              f"was never trained to match against a fixed gold vocabulary)")

    unload_model(model)
    print(f"Done. Results in {save_folder}")


if __name__ == "__main__":
    main()

