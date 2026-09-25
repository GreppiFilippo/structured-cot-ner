"""CoT distillation SFT training for MedGemma-4B — standalone script for
SLURM/Docker, converted from MedGemma_CoT_SFT.ipynb. Same logic
throughout (config, data loading, length measurement, LoRA, trainer,
save/push) — only the environment (no Colab/Kaggle secrets, plain env
vars for HF_TOKEN) and entry point (argparse instead of notebook cells)
differ.

Usage:
    python3 train_sft.py \\
        --data-path data/sft_ready_filtered_v3.jsonl \\
        --hf-repo-id filippogreppi/MedGemma-CoT-distilled
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import datasets
import numpy as np
import peft
import torch
import transformers
import trl


def measure_max_seq_length(dataset, processor, percentile: int = 99) -> int:
    """Measures real token lengths on the actual training data, instead
    of guessing a fixed max_length — the earlier label-only experiment
    hardcoded max_length=2048, which is far too short for CoT reasoning
    traces (the prompt alone runs ~1400+ tokens, plus a full reasoning
    trace in the assistant turn)."""
    lengths = []
    for row in dataset:
        text = processor.apply_chat_template(
            row["messages"], add_generation_prompt=False, tokenize=False)
        n_tokens = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n_tokens)

    lengths = np.array(lengths)
    print(f"Lengths (tokens) over {len(lengths)} training examples:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p}: {int(np.percentile(lengths, p))}")
    print(f"  mean: {lengths.mean():.0f}")

    p_val = int(np.percentile(lengths, percentile))
    max_seq_length = 1 << (p_val - 1).bit_length()  # next power of 2
    n_truncated = int((lengths > max_seq_length).sum())
    print(f"\nMAX_SEQ_LENGTH chosen: {max_seq_length} (covers p{percentile}={p_val})")
    print(f"Examples above this threshold (will be truncated): {n_truncated} out of {len(lengths)}")
    return max_seq_length


def collate_fn(samples, processor, max_length):
    """Collate SFT samples into a padded batch with masked labels.
    Labels before the first <start_of_turn>model token are masked to
    -100 so the loss is computed only on assistant turns (plus the two
    structural delimiter tokens themselves — same behavior as the
    earlier label-only experiment's own collate_fn, verified by test)."""
    texts = [
        processor.apply_chat_template(
            s["messages"], add_generation_prompt=False, tokenize=False).strip()
        for s in samples
    ]
    batch = processor(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length, add_special_tokens=False)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    start_id = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
    model_id = processor.tokenizer.convert_tokens_to_ids("model")
    for i, ids in enumerate(batch["input_ids"]):
        for j in range(len(ids) - 1):
            if ids[j] == start_id and ids[j + 1] == model_id:
                labels[i, :j] = -100
                break

    batch["labels"] = labels
    return batch


def execute_sft(model, processor, ds, max_seq_length, config, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ds["train"]
    if config.n_train_samples:
        train_ds = train_ds.select(range(min(config.n_train_samples, len(train_ds))))
    eval_ds = ds["validation"]
    print(f"Training on {len(train_ds)} examples, validation on {len(eval_ds)}")

    lora_cfg = peft.LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        r=config.lora_r,
        # language model only: a plain list of names also matches the
        # vision tower's q/k/v_proj, whose LoRA weights stay at zero in
        # text-only training (confirmed on the published adapters: 81/81
        # vision lora_B all-zero) — dead parameters, no effect on outputs
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        task_type="CAUSAL_LM",
    )

    est_steps = max(1, len(train_ds) // (config.per_device_batch * config.grad_accum_steps)) * config.num_epochs
    effective_save = min(config.save_steps, max(1, est_steps // 3))
    print(f"Estimated steps: {est_steps}, eval/save every: {effective_save}")

    # 2026-09-15: build the SFTConfig kwargs as a dict first, then filter
    # against what this installed trl version's SFTConfig actually
    # accepts, instead of passing them all directly — a real crash
    # ("SFTConfig.__init__() got an unexpected keyword argument
    # 'warmup_ratio'") showed the installed trl version's SFTConfig
    # doesn't accept every standard TrainingArguments field, likely from
    # pip's resolver picking an older trl to satisfy some other
    # dependency's constraint (trl was deliberately left unpinned, same
    # "upgrade to latest" reasoning used elsewhere in this project — but
    # unlike transformers, trl's own version compatibility here wasn't
    # independently verified before this first real run). Dropping
    # unsupported kwargs with a warning keeps the run going instead of
    # crashing on the next one encountered.
    import inspect
    sft_kwargs = dict(
        output_dir=str(ckpt_dir),
        seed=config.random_seed,
        data_seed=config.random_seed,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch,
        per_device_eval_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=config.learning_rate,
        bf16=True,
        fp16=False,
        max_grad_norm=0.1,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        logging_steps=10,
        save_strategy="steps",
        save_steps=effective_save,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=effective_save,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=os.environ.get("SFT_REPORT_TO", "none"),
        run_name="MedGemma-CoT-distilled",
    )
    accepted_params = set(inspect.signature(trl.SFTConfig.__init__).parameters)
    dropped = {k: v for k, v in sft_kwargs.items() if k not in accepted_params}
    if dropped:
        print(f"[warn] This installed version of trl.SFTConfig does not support these "
              f"parameters, they will be ignored: {sorted(dropped.keys())}")
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in accepted_params}

    sft_args = trl.SFTConfig(**sft_kwargs)

    trainer = trl.SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_cfg,
        processing_class=processor,
        data_collator=lambda samples: collate_fn(samples, processor, max_seq_length),
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=5)],
    )

    existing_ckpts = sorted(ckpt_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
    resume = existing_ckpts[-1] if existing_ckpts else False
    if resume:
        print(f"Resuming from checkpoint: {resume}")
    else:
        print("No checkpoint found — training from scratch.")

    trainer.train(resume_from_checkpoint=resume)
    return trainer


def save_adapter(trainer, processor, adapter_dir: Path, processor_dir: Path):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(processor_dir))

    adapter_files = list(adapter_dir.iterdir())
    print(f"\nFiles in adapter/: {[f.name for f in adapter_files]}")
    assert (adapter_dir / "adapter_config.json").exists(), (
        "FATAL: adapter_config.json missing — adapter NOT saved correctly!")
    safetensors = [f for f in adapter_files if "adapter_model" in f.name]
    assert safetensors, (
        "FATAL: adapter_model.safetensors missing — adapter NOT saved correctly!")
    size_mb = safetensors[0].stat().st_size / 1e6
    assert size_mb < 500, (
        f"FATAL: adapter is {size_mb:.0f} MB — the full model was saved instead of an adapter!")
    print(f"Adapter saved correctly ({size_mb:.1f} MB)")


def push_to_hub(adapter_dir: Path, processor_dir: Path, repo_id: str, hf_token: str, base_model_id: str):
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=hf_token)

    create_repo(repo_id, token=hf_token, exist_ok=True, private=True)
    print(f"Repo: https://huggingface.co/{repo_id}")

    readme = f"""---
tags:
  - biomedical
  - ner
  - lora
  - medgemma
  - chain-of-thought
base_model: {base_model_id}
library_name: peft
---

# MedGemma-CoT-distilled

LoRA adapter fine-tuned on a jury-filtered Chain-of-Thought NER dataset
(reasoning + open-schema entity extraction, distilled from Qwen3-32B via
a three-judge validation pipeline). Load with:

```python
import peft, transformers, torch
base = transformers.AutoModelForImageTextToText.from_pretrained(
    "{base_model_id}", device_map="auto", dtype=torch.bfloat16)
model = peft.PeftModel.from_pretrained(base, "{repo_id}")
model = model.merge_and_unload()
```
"""
    api.upload_file(
        path_or_fileobj=readme.encode(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="model", token=hf_token,
    )
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model", token=hf_token)
    api.upload_folder(folder_path=str(processor_dir), repo_id=repo_id, repo_type="model", token=hf_token)
    print("Upload complete.")


class Config:
    def __init__(self, args):
        self.random_seed = 42
        self.n_train_samples = args.n_train_samples
        self.num_epochs = args.num_epochs
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.per_device_batch = args.per_device_batch
        self.grad_accum_steps = args.grad_accum_steps
        self.learning_rate = args.learning_rate
        self.save_steps = args.save_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="sft_ready_filtered_v3.jsonl (or whichever filtered dataset)")
    parser.add_argument("--base-model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--hf-repo-id", default="filippogreppi/MedGemma-CoT-distilled")
    parser.add_argument("--work-dir", default="./work")
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--n-train-samples", type=int, default=None, help="Cap on training rows; omit to use all")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--skip-push", action="store_true", help="Save the adapter locally but don't push to HF Hub")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.skip_push:
        raise SystemExit("HF_TOKEN environment variable is required (unless --skip-push is passed).")

    report_to = os.environ.get("SFT_REPORT_TO", "none")
    if report_to == "wandb":
        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            raise SystemExit(
                "SFT_REPORT_TO=wandb but WANDB_API_KEY is not set — "
                "pass it through run_docker_sft.sh (already forwarded via -e) "
                "or set --report-to none to skip logging."
            )
        import wandb
        wandb.login(key=wandb_key)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "medgemma-cot-sft"),
            name="MedGemma-CoT-distilled",
        )
        print(f"wandb active — project: {os.environ.get('WANDB_PROJECT', 'medgemma-cot-sft')}")

    config = Config(args)
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"
    adapter_dir = work_dir / "adapter"
    processor_dir = work_dir / "processor"

    # --- 1. data ---
    raw = datasets.load_dataset("json", data_files=args.data_path)["train"]
    print(f"{len(raw)} examples loaded from {args.data_path}")
    split = raw.train_test_split(test_size=args.validation_size, seed=config.random_seed)
    cot_ds = datasets.DatasetDict({"train": split["train"], "validation": split["test"]})
    print(f"Train: {len(cot_ds['train'])}  Validation: {len(cot_ds['validation'])}")

    # --- 2. tokenizer for length measurement (before loading the full model) ---
    measure_processor = transformers.AutoProcessor.from_pretrained(args.base_model_id, use_fast=True)
    max_seq_length = measure_max_seq_length(cot_ds["train"], measure_processor)
    del measure_processor

    # --- 3. model ---
    processor = transformers.AutoProcessor.from_pretrained(args.base_model_id, use_fast=True)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.base_model_id, device_map="auto", dtype=torch.bfloat16,
    )
    model.eval()
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    # --- 4. training ---
    trainer = execute_sft(model, processor, cot_ds, max_seq_length, config, ckpt_dir)

    # --- 5. saving ---
    save_adapter(trainer, processor, adapter_dir, processor_dir)

    # --- 6. upload to HF ---
    if not args.skip_push:
        push_to_hub(adapter_dir, processor_dir, args.hf_repo_id, hf_token, args.base_model_id)

        # --- 7. round-trip verification ---
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print("\nRound-trip verification: reloading the adapter from HF...")
        base_model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.base_model_id, device_map="auto", dtype=torch.bfloat16)
        ft_model = peft.PeftModel.from_pretrained(base_model, args.hf_repo_id, token=hf_token)
        ft_model = ft_model.merge_and_unload()
        ft_model.eval()

        prompt = cot_ds["validation"][0]["messages"][:-1]
        text = processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt", add_special_tokens=False).to(ft_model.device)
        with torch.inference_mode():
            out = ft_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        reply = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n=== ROUND-TRIP VERIFICATION ===")
        print(reply[:500])

    print("\nDone.")


if __name__ == "__main__":
    main()