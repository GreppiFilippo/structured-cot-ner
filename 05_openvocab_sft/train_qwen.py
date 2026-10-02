"""LoRA SFT of a Qwen3 model (thinking mode) on the open-vocabulary dataset
built by build_openvocab_dataset.py.

The same script runs on the cluster (Qwen3-4B, bf16, RTX 3090) and on a
small local GPU for pilots (e.g. Qwen3-1.7B in bf16, or Qwen3-4B with
--load-in-4bit on an 8 GB card): only the flags change.

Loss masking: each example is rendered with the model's own chat template;
the generation prompt (system + user + "<|im_start|>assistant\\n") is an
exact prefix of the rendered example, and everything up to it is masked,
so the loss covers "<think>reasoning</think>" + JSON + <|im_end|>.
Examples longer than --max-seq-length are dropped, not truncated — a
truncated example would teach the model to stop before the JSON.

Pilot on an 8 GB GPU:
    python train_qwen.py --data-path data/openvocab_train.jsonl \\
        --model-id Qwen/Qwen3-1.7B --n-train-samples 1000 --num-epochs 1 \\
        --output-dir work/pilot --skip-push

Cluster:
    python3 train_qwen.py --data-path data/openvocab_train.jsonl \\
        --model-id Qwen/Qwen3-4B --hf-repo-id yourname/Qwen3-4B-openvocab-cot
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset


def resolve_data_path(path: str) -> str:
    """A local path, or hf://<owner>/<dataset-repo>/<file> for a file in a
    (private) HF dataset repo — downloaded first, so cluster jobs can read
    data that lives on no shared filesystem."""
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download
    owner, repo, filename = path[len("hf://"):].split("/", 2)
    return hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


def render(tokenizer, messages):
    """(input_ids, labels) for one example, loss only on the assistant turn."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the generation prompt is not a prefix of the full example")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full.rstrip("\n"), add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenization of the prompt differs inside the full example")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def build_split(tokenizer, rows, max_len):
    ids, labels, dropped = [], [], 0
    for row in rows:
        i, l = render(tokenizer, row["messages"])
        if len(i) > max_len:
            dropped += 1
            continue
        ids.append(i)
        labels.append(l)
    return {"input_ids": ids, "labels": labels}, dropped


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="Local .jsonl, or hf://<owner>/<dataset-repo>/<file>")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="work/qwen3-openvocab")
    parser.add_argument("--load-in-4bit", action="store_true", help="QLoRA (bitsandbytes NF4) — for small GPUs")
    parser.add_argument("--n-train-samples", type=int, default=None, help="Cap on training rows (pilots)")
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Default: 99th percentile of the rendered lengths, rounded up to a multiple of 256")
    parser.add_argument("--num-epochs", type=float, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke tests)")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU (smoke tests on a machine without CUDA)")
    args = parser.parse_args()
    transformers.set_seed(args.seed)

    import peft

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    raw = load_dataset("json", data_files=resolve_data_path(args.data_path))["train"]
    split = raw.train_test_split(test_size=args.validation_size, seed=args.seed)
    train_rows, val_rows = split["train"], split["test"]
    if args.n_train_samples:
        train_rows = train_rows.select(range(min(args.n_train_samples, len(train_rows))))
        val_rows = val_rows.select(range(min(max(args.n_train_samples // 20, 20), len(val_rows))))

    max_len = args.max_seq_length
    if max_len is None:
        lengths = np.array([len(render(tokenizer, r["messages"])[0]) for r in train_rows])
        p99 = int(np.percentile(lengths, 99))
        max_len = int(np.ceil(p99 / 256) * 256)
        print(f"rendered lengths: p50={int(np.percentile(lengths, 50))} p95={int(np.percentile(lengths, 95))} "
              f"p99={p99} max={lengths.max()} -> max_seq_length={max_len}")
    train_data, dropped_train = build_split(tokenizer, train_rows, max_len)
    val_data, dropped_val = build_split(tokenizer, val_rows, max_len)
    print(f"train: {len(train_data['input_ids'])} examples ({dropped_train} longer than {max_len} dropped), "
          f"validation: {len(val_data['input_ids'])} ({dropped_val} dropped)")
    from datasets import Dataset
    train_ds, val_ds = Dataset.from_dict(train_data), Dataset.from_dict(val_data)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    model_kwargs = {"dtype": torch.bfloat16 if bf16 else torch.float32}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    if use_cuda:
        model_kwargs["device_map"] = {"": 0}
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    proj = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # multimodal checkpoints (e.g. MedGemma = Gemma3ForConditionalGeneration) also have q/k/v
    # projections in the vision tower, which text-only training would never update: restrict
    # LoRA to the language model there. Text-only models (Qwen3) keep the plain module list.
    is_multimodal = any("vision" in n for n, _ in model.named_modules())
    target = rf".*language_model.*\.({'|'.join(proj)})" if is_multimodal else proj
    lora = peft.LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                           target_modules=target)
    model = peft.get_peft_model(model, lora)
    model.print_trainable_parameters()

    out_dir = Path(args.output_dir)
    training_args = transformers.TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        use_cpu=args.use_cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * len(train_ds) * args.num_epochs / (args.per_device_batch * args.grad_accum_steps))),
        bf16=bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=os.environ.get("SFT_REPORT_TO", "none"),
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=0,  # Windows-safe
    )
    trainer = transformers.Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                                   data_collator=Collator(tokenizer.pad_token_id))
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "training_args.json").write_text(json.dumps(vars(args) | {"max_seq_length": max_len}, indent=2))
    print(f"adapter saved to {adapter_dir}")

    if args.hf_repo_id and not args.skip_push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.hf_repo_id, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(adapter_dir), repo_id=args.hf_repo_id, repo_type="model")
        print(f"pushed to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
