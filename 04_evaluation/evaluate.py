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
    python3 evaluate.py --model-type cot_typed --dataset-name ncbi --max-new-tokens 5120

    # any causal LM (e.g. Qwen3) with the baseline's closed-schema prompt,
    # optionally with a LoRA adapter; results in outputs/<dataset>/<run-name>/
    python3 evaluate.py --model-type causal --model-id Qwen/Qwen3-4B --run-name qwen3-4b_base \
        --dataset-name ncbi --max-new-tokens 8192 --do-sample
    python3 evaluate.py --model-type causal --model-id Qwen/Qwen3-4B --adapter yourname/Qwen3-4B-openvocab-cot \
        --run-name qwen3-4b_sft --dataset-name ncbi --max-new-tokens 8192 --do-sample
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path

import datasets
import peft
import torch
import transformers
from tqdm.auto import tqdm

from src.metrics import parse_gold, score_run, write_results
from src.prompts import FreeCoTPromptBuilder

BASE_MODEL_ID = "google/medgemma-1.5-4b-it"
PRETRAIN_PATH = "filippogreppi/MedGemma-pretrain"
COT_DISTILLED_PATH = "filippogreppi/MedGemma-CoT-distilled"
SUBSET_REPO_ID_DEFAULT = "filippogreppi/bioner-eval-subset"

# output subfolder per model type; "pre_sft"/"post_sft" are kept as-is
# because resume-skip and the aggregate scripts look them up by name
SUBFOLDERS = {"baseline": "pre_sft", "pretrain": "post_sft", "cot": "post_sft_cot", "cot_typed": "post_sft_cot_typed"}

_PADDING_TOKEN_RE = re.compile(r"<(?:pad|eos|bos)>|<\|endoftext\|>")
_END_OF_TURN_RE = re.compile(r"<end_of_turn>|<\|im_end\|>")


class BatchInferencer:
    def __init__(self, model, processor, max_new_tokens=512, generation_kwargs=None, template_kwargs=None):
        """processor: MedGemma's AutoProcessor or a plain tokenizer (causal LMs).
        template_kwargs: extra chat-template arguments, e.g. {"enable_thinking": False}.
        generation_kwargs: defaults to greedy decoding (the setting of every
        MedGemma run); pass sampling parameters for thinking models."""
        self._model, self._processor = model, processor
        self._max_new_tokens = max_new_tokens
        self._generation_kwargs = generation_kwargs or {"do_sample": False}
        self._template_kwargs = template_kwargs or {}
        self._tok = tok = getattr(processor, "tokenizer", processor)
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
            self._processor.apply_chat_template(p, add_generation_prompt=True, tokenize=False, **self._template_kwargs)
            for p in prompts
        ]
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
        ).to(self._model.device)

        try:
            with torch.inference_mode():
                outputs = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens,
                                               pad_token_id=self._tok.pad_token_id, **self._generation_kwargs)
            input_len = inputs.input_ids.shape[1]
            new_tokens = outputs[:, input_len:]
            replies = self._tok.batch_decode(new_tokens, skip_special_tokens=False)
            return [_PADDING_TOKEN_RE.sub("", _END_OF_TURN_RE.split(r)[0]).strip() for r in replies]
        finally:
            del inputs
            if "outputs" in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VLLMInferencer:
    """Same interface as BatchInferencer, generation with vLLM instead of
    transformers (continuous batching: every row of a chunk is generated
    concurrently, which for thinking models with long, uneven replies is
    one to two orders of magnitude faster). Prompts are rendered with the
    same chat template, and replies are post-processed the same way, so
    parsing and scoring are unchanged. Outputs differ slightly from the
    transformers engine (different kernels and random streams): compare
    models evaluated with the same engine.

    A LoRA adapter is applied by vLLM itself (LoRARequest), after checking
    its weights are non-zero, like check_adapter_applied does."""

    def __init__(self, model_id, adapter=None, max_new_tokens=4096, generation_kwargs=None, seed=0,
                 hf_token=None, max_model_len=None, template_kwargs=None):
        from vllm import LLM, SamplingParams

        self._tok = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self._template_kwargs = template_kwargs or {}
        self._lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            path = adapter if os.path.isdir(adapter) else _download_adapter(adapter, hf_token)
            _check_adapter_files(path)
            self._lora = LoRARequest("adapter", 1, path)
            rank = json.loads((Path(path) / "adapter_config.json").read_text())["r"]
        self._llm = LLM(model=model_id, dtype="bfloat16", seed=seed, enable_lora=bool(adapter),
                        max_lora_rank=max(16, rank) if adapter else 16,
                        max_model_len=max_model_len or max_new_tokens + 4096, gpu_memory_utilization=0.9)
        gk = generation_kwargs or {"do_sample": False}
        if gk.get("do_sample"):
            self._params = SamplingParams(max_tokens=max_new_tokens, temperature=gk["temperature"],
                                          top_p=gk["top_p"], top_k=gk["top_k"], skip_special_tokens=False)
        else:
            self._params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, skip_special_tokens=False)

    def disable_gradient_checkpointing(self):
        pass

    def generate(self, prompts):
        texts = [self._tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False, **self._template_kwargs)
                 for p in prompts]
        # The rendered template already starts with the BOS token where the model has one (Gemma:
        # "<bos><start_of_turn>..."); a text prompt would be re-tokenized by vLLM WITH special
        # tokens, doubling it — Gemma then answers very differently (MedGemma stopped reasoning).
        # Token ids without added special tokens = exactly what the transformers engine feeds.
        token_prompts = [{"prompt_token_ids": self._tok(t, add_special_tokens=False)["input_ids"]} for t in texts]
        outputs = self._llm.generate(token_prompts, self._params, lora_request=self._lora, use_tqdm=False)
        return [_PADDING_TOKEN_RE.sub("", _END_OF_TURN_RE.split(o.outputs[0].text)[0]).strip() for o in outputs]


def _download_adapter(repo_id, hf_token):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, token=hf_token)


def _check_adapter_files(path):
    """check_adapter_applied for an adapter that vLLM loads itself."""
    from safetensors.torch import load_file

    weights = load_file(str(Path(path) / "adapter_model.safetensors"))
    lora_b = [t for k, t in weights.items() if "lora_B" in k]
    trained = sum(int(t.abs().max().item() > 0) for t in lora_b)
    print(f"  [adapter check] {path}: {trained}/{len(lora_b)} LoRA B matrices non-zero")
    if not trained:
        raise SystemExit(f"adapter {path}: every LoRA B matrix is zero — nothing was trained or loaded")


# --- Model loading ---


def load_baseline():
    processor = transformers.AutoProcessor.from_pretrained(BASE_MODEL_ID, backend="torchvision")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model.eval()
    return model, processor


def check_adapter_applied(peft_model, name):
    """Fails loudly if the LoRA weights did not actually load. When the
    module names saved in an adapter do not match the loaded model (e.g. a
    different transformers version renamed them), PEFT only warns and
    leaves every lora_B at its zero initialization — the run would then
    silently evaluate the base model under the adapter's name."""
    lora_b = [p for n, p in peft_model.named_parameters() if "lora_B" in n]
    trained = sum(int(p.detach().abs().max().item() > 0) for p in lora_b)
    print(f"  [adapter check] {name}: {trained}/{len(lora_b)} LoRA B matrices non-zero "
          f"(the rest, if any, are modules never used in text-only training, e.g. the vision tower)")
    if not trained:
        raise SystemExit(f"adapter {name}: no LoRA weight was loaded — module names do not match the model")


def _load_adapter(adapter_path, hf_token, verbose):
    processor = transformers.AutoProcessor.from_pretrained(BASE_MODEL_ID, backend="torchvision")
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto", dtype=torch.bfloat16)
    model = peft.PeftModel.from_pretrained(base, adapter_path, token=hf_token)
    check_adapter_applied(model, adapter_path)
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


def load_causal(model_id, adapter=None, hf_token=None, load_in_4bit=False):
    """Any causal LM (e.g. Qwen3), optionally with a LoRA adapter merged in."""
    kwargs = {"device_map": "auto", "dtype": torch.bfloat16, "token": hf_token}
    if load_in_4bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(adapter or model_id, token=hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        model = peft.PeftModel.from_pretrained(model, adapter, token=hf_token)
        check_adapter_applied(model, adapter)
        if not load_in_4bit:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Evaluation loop ---


def requested_types(row):
    """The entity types the closed-schema prompt of this row asks for —
    exactly the keys of its gold answer, in prompt order."""
    return list(parse_gold(row["messages"][-1]["content"]).keys())


def add_type_descriptions(messages, descriptions):
    """Appends a definition of each requested entity type to the user turn (closed-schema prompt),
    right after the ENTITY TYPES list; types without a description are listed as-is."""
    user = messages[1]["content"]
    types = json.loads(user.split("ENTITY TYPES:\n", 1)[1].strip())
    if not types:
        return messages
    lines = "\n".join(f"- {t}: {descriptions.get(t, '(no definition)')}" for t in types)
    user = user.rstrip("\n") + f"\n\nENTITY TYPE DEFINITIONS:\n{lines}\n"
    return [messages[0], {"role": "user", "content": user}]


def build_prompts(batch, model_type, cot_prompt_builder, type_descriptions=None):
    if model_type == "cot":
        return [[{"role": "user", "content": cot_prompt_builder.build(row["raw_text"])}] for row in batch]
    if model_type == "cot_typed":
        return [[{"role": "user", "content": cot_prompt_builder.build_typed(row["raw_text"], requested_types(row))}]
                for row in batch]
    if type_descriptions:
        return [add_type_descriptions(row["messages"][:-1], type_descriptions) for row in batch]
    return [row["messages"][:-1] for row in batch]


def evaluate(model, processor, test_data, save_folder, model_type, batch_size=4, max_new_tokens=512,
             generation_kwargs=None, inferencer=None, template_kwargs=None, type_descriptions=None):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    if inferencer is None:
        inferencer = BatchInferencer(model=model, processor=processor, max_new_tokens=max_new_tokens,
                                     generation_kwargs=generation_kwargs, template_kwargs=template_kwargs)
    inferencer.disable_gradient_checkpointing()

    # "cot" / "cot_typed" differ from the closed-schema models in their
    # prompt (built from raw_text; "cot_typed" adds the requested types)
    # and their output parser; "cot" is also scored against the full gold
    # — see src/metrics.select_gold and score_run.
    cot_prompt_builder = FreeCoTPromptBuilder() if model_type.startswith("cot") else None

    # Raw generations are appended to generations_partial.jsonl after every
    # batch, so an interrupted run (job time limit, machine switched off)
    # resumes from the first row not generated yet instead of starting over.
    # With sampling, resumed rows use a different random stream than an
    # uninterrupted run would have — rows are still generated exactly once.
    partial_path = save_folder / "generations_partial.jsonl"
    done = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["idx"]] = rec["pred_raw"]
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut short by the interruption: that row is regenerated
        print(f"  resuming: {len(done)} rows already generated in {partial_path.name}")

    n = len(test_data)
    todo = [i for i in range(n) if i not in done]
    with open(partial_path, "a", encoding="utf-8") as partial:
        for start in tqdm(range(0, len(todo), batch_size), desc=save_folder.name, leave=False):
            idxs = todo[start:start + batch_size]
            batch = [test_data[i] for i in idxs]
            preds_raw = inferencer.generate(build_prompts(batch, model_type, cot_prompt_builder, type_descriptions))
            for i, pred_raw in zip(idxs, preds_raw):
                done[i] = pred_raw
                partial.write(json.dumps({"idx": i, "pred_raw": pred_raw}, ensure_ascii=False) + "\n")
            partial.flush()

    items = [(i, test_data[i], done[i]) for i in range(n)]
    scorers, records = score_run(items, model_type)
    metrics = write_results(save_folder, scorers, records)
    partial_path.unlink()  # everything is in predictions.jsonl now
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", choices=list(SUBFOLDERS) + ["causal", "medgemma"], required=True,
                         help="cot_typed = the CoT-distilled model with the requested entity types added "
                              "to its prompt (see src/prompts.TYPED_SECTION), scored like baseline/pretrain")
    parser.add_argument("--dataset-name", required=True,
                         help="One of: ncbi, bc5cdr, bc2gm, bc4chemd, AnatEM, JNLPBA, "
                              "JNLPBA-Rare, MedMentions-Rare, biored")
    parser.add_argument("--subset-repo-id", default=SUBSET_REPO_ID_DEFAULT)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rows (pilots / smoke tests)")
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
    causal = parser.add_argument_group(
        "--model-type causal (any causal LM) / medgemma (MedGemma with any --adapter): closed-schema prompt "
        "like the baseline, results in outputs/<dataset>/<run-name>/")
    causal.add_argument("--model-id", default="Qwen/Qwen3-4B")
    causal.add_argument("--adapter", default=None, help="LoRA adapter (HF repo or local folder)")
    causal.add_argument("--run-name", default=None, help="Output subfolder, e.g. qwen3-4b_base")
    causal.add_argument("--load-in-4bit", action="store_true", help="For small GPUs — not comparable with bf16 runs")
    causal.add_argument("--do-sample", action="store_true",
                        help="Sample instead of greedy decoding. Qwen3 in thinking mode should not be run greedy "
                             "(its model card warns of endless repetitions); uses its recommended "
                             "temperature/top_p/top_k below")
    causal.add_argument("--temperature", type=float, default=0.6)
    causal.add_argument("--top-p", type=float, default=0.95)
    causal.add_argument("--top-k", type=int, default=20)
    causal.add_argument("--seed", type=int, default=0)
    causal.add_argument("--engine", choices=["transformers", "vllm"], default="transformers",
                        help="Generation engine for --model-type causal or medgemma. vllm is much faster for thinking "
                             "models; --batch-size is then the number of rows generated together (and "
                             "saved for resuming) per chunk, e.g. 100.")
    causal.add_argument("--no-thinking", action="store_true",
                        help="Qwen3: render the prompt with enable_thinking=False (the template pre-fills an empty "
                             "<think></think>, so the model answers directly) — zero-shot without native reasoning")
    causal.add_argument("--max-model-len", type=int, default=None,
                        help="vllm only: prompt + reply token budget (default: max-new-tokens + 4096)")
    parser.add_argument("--type-descriptions", default=None,
                        help="JSON {entity type: definition}: appended to the prompt after the ENTITY TYPES list "
                             "(closed-schema prompt only); scoring is unchanged")
    parser.add_argument("--allow-missing-all-gold", action="store_true",
                         help="For --model-type cot only: run even if the subset has no 'all_gold' column, "
                              "scoring against the closed-schema prompt's gold instead. That gold omits every "
                              "entity type not sampled into the row's prompt, so the CoT model's precision "
                              "comes out artificially low — only use this for debugging.")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    named_run = args.model_type in ("causal", "medgemma")
    if named_run and not args.run_name:
        raise SystemExit(f"--model-type {args.model_type} needs --run-name (the output subfolder)")
    subfolder = args.run_name if named_run else SUBFOLDERS[args.model_type]
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
    if args.limit:
        test_data = test_data.select(range(min(args.limit, len(test_data))))
    print(f"  {len(test_data)} rows loaded.")

    if (args.model_type == "cot" and "all_gold" not in test_data.column_names
            and not args.allow_missing_all_gold):
        raise SystemExit(
            "The eval subset has no 'all_gold' column, which the CoT model needs as gold "
            "(see src/metrics.select_gold). Rebuild it first with build_eval_subset.py "
            "(prompts stay identical, checked row by row before pushing), or pass "
            "--allow-missing-all-gold to run anyway.")

    print(f"Loading {args.model_type} model...")
    generation_kwargs = None
    if args.do_sample:
        if args.engine != "vllm":  # vLLM seeds itself; initializing CUDA here would break its engine process
            transformers.set_seed(args.seed)
        generation_kwargs = {"do_sample": True, "temperature": args.temperature,
                             "top_p": args.top_p, "top_k": args.top_k}
    inferencer = None
    template_kwargs = {"enable_thinking": False} if args.no_thinking else None
    if args.engine == "vllm":
        if args.model_type not in ("causal", "medgemma"):
            raise SystemExit("--engine vllm is only supported with --model-type causal or medgemma")
        model, processor = None, None
        # medgemma: MedGemma-1.5 itself (vLLM serves its language model and any LoRA on it)
        vllm_model_id = BASE_MODEL_ID if args.model_type == "medgemma" else args.model_id
        inferencer = VLLMInferencer(vllm_model_id, args.adapter, args.max_new_tokens, generation_kwargs,
                                    args.seed, hf_token, args.max_model_len, template_kwargs)
    elif args.model_type == "causal":
        model, processor = load_causal(args.model_id, args.adapter, hf_token, args.load_in_4bit)
    elif args.model_type == "medgemma":
        model, processor = (_load_adapter(args.adapter, hf_token, verbose=True) if args.adapter
                            else load_baseline())
    elif args.model_type == "baseline":
        model, processor = load_baseline()
    else:
        if not hf_token:
            raise SystemExit(f"--hf-token (or HF_TOKEN env var) is required for --model-type "
                             f"{args.model_type} (needed to pull the private LoRA adapter).")
        loader = load_pretrain if args.model_type == "pretrain" else load_cot_distilled
        model, processor = loader(hf_token, verbose=args.model_type.startswith("cot"))

    print(f"Evaluating on {args.dataset_name} ({args.model_type})...")
    # a causal LM gets the baseline's closed-schema prompt and is scored like it
    scoring_type = "baseline" if named_run else args.model_type
    metrics = evaluate(model, processor, test_data, save_folder, model_type=scoring_type,
                       batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                       generation_kwargs=generation_kwargs, inferencer=inferencer,
                       template_kwargs=template_kwargs,
                       type_descriptions=json.loads(Path(args.type_descriptions).read_text())
                       if args.type_descriptions else None)
    print(f"F1_type_agnostic={metrics['F1_micro_type_agnostic']:.4f}  "
          f"P={metrics['precision_type_agnostic']:.4f}  R={metrics['recall_type_agnostic']:.4f}  "
          f"n={metrics['n_samples']}")
    # type-exact F1 is near zero for the CoT model by construction: it
    # names its own types, never matched against the gold vocabulary
    print(f"F1_type_exact={metrics['F1_micro']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
    if "requested_types_only" in metrics:
        m = metrics["requested_types_only"]
        print(f"requested types only: F1_type_agnostic={m['F1_micro_type_agnostic']:.4f}  "
              f"F1_type_exact={m['F1_micro']:.4f}")
    print(f"parse_status_counts={metrics['parse_status_counts']}  gold_source={metrics['gold_source']}")

    if model is not None:
        unload_model(model)
    print(f"Done. Results in {save_folder}")


if __name__ == "__main__":
    main()
