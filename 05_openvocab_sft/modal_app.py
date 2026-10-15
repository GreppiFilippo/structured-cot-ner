"""Runs the v3 open-vocabulary experiment on Modal (serverless GPUs):
LoRA SFT of a Qwen3 model with train_qwen.py, and evaluation of the base
and fine-tuned model on the 9 BioNER benchmarks with evaluate.py (vLLM
engine, sampling as recommended for Qwen3 thinking mode). Same code, data
and metric as the local and cluster runs; only where it runs changes.

Setup (once): `modal setup`, and a Modal secret named "huggingface" with
HF_TOKEN (write access: training pushes the adapter to the Hub).

Usage, from the repository root:
    # a few cents: 3 training steps on Qwen3-1.7B + 10 rows of ncbi
    .venv/bin/modal run 05_openvocab_sft/modal_app.py::smoke

    # the full experiment, detached (the Mac can be switched off)
    .venv/bin/modal run --detach 05_openvocab_sft/modal_app.py::full

    # results (same layout as 04_evaluation/outputs/)
    .venv/bin/modal volume get tesi-outputs eval 04_evaluation/outputs_modal

Every artefact lives in the "tesi-outputs" volume: eval/<dataset>/<run>/
(metrics, predictions), work/<tag>/ (checkpoints, adapter), env/ (exact
package versions of the image, for reproducibility).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
CODE = "/root/code"
# per-container cap for evaluate_run: a stuck job is billed until it hits this.
# Short evaluations (e.g. CrossNER) can lower it: EVAL_TIMEOUT_S=1800 modal run ...
EVAL_TIMEOUT_S = int(os.environ.get("EVAL_TIMEOUT_S", 8 * 3600))
GPU = "L40S"  # 48 GB, faster than an RTX 3090; ~2 $/h

DATASETS = ["ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"]
HF_DATA = "hf://filippogreppi/openvocab-cot-sft/openvocab_train_v3_qwen.jsonl"
# the 9 benchmarks + the in-domain Pile-NER test (its own subset repo)
ALL = DATASETS + ["pilener"]
PILENER = "filippogreppi/pilener-eval-subset"

# CUDA *devel* base (nvcc): vLLM JIT-compiles some kernels at startup. Its
# CUDA version must match the torch wheel vLLM pulls in (cu130 for 0.30.0).
# Versions pinned for reproducibility; the full list of every run's image is
# also saved to /outputs/env/.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm==0.30.0", "transformers==5.17.0", "peft==0.21.0", "datasets", "accelerate",
                 "scipy", "numpy", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_dir(REPO / "04_evaluation", f"{CODE}/04_evaluation",
                   ignore=["outputs*", "__pycache__", "slurm_logs", "*.jsonl", ".env"])
    .add_local_dir(REPO / "05_openvocab_sft", f"{CODE}/05_openvocab_sft",
                   ignore=["data", "work", "__pycache__", ".env"])
)
# training files of the free vs structured CoT experiment (build_tutor_cot_dataset.py; not in git,
# derived from the tutors' private dataset): added only where they have been built
for _v in ("free", "structured"):
    _f = REPO / f"05_openvocab_sft/data/pilener_tutor_{_v}_cot.jsonl"
    if _f.exists():
        image = image.add_local_file(_f, f"{CODE}/tutor_data/{_v}.jsonl")
hf_cache = modal.Volume.from_name("tesi-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("tesi-outputs", create_if_missing=True)
VOLUMES = {"/cache": hf_cache, "/outputs": outputs}
SECRETS = [modal.Secret.from_name("huggingface")]

app = modal.App("tesi-openvocab", image=image)


def _run(cmd: list[str], cwd: str) -> None:
    """Runs a script, committing the outputs volume every 10 minutes so
    checkpoints and partial generations survive a preempted container
    (both scripts resume from them when re-run)."""
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(600):
            outputs.commit()

    threading.Thread(target=commit_loop, daemon=True).start()
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    finally:
        stop.set()
        outputs.commit()
        hf_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=24 * 3600)
def train(model_id: str, data_path: str, tag: str, hf_repo_id: str | None, extra: list[str] = ()) -> str:
    Path("/outputs/env").mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    Path(f"/outputs/env/pip_freeze_{tag}.txt").write_text(freeze)
    cmd = ["python", "train_qwen.py", "--model-id", model_id, "--data-path", data_path,
           "--output-dir", f"/outputs/work/{tag}", *extra]
    cmd += ["--hf-repo-id", hf_repo_id] if hf_repo_id else ["--skip-push"]
    _run(cmd, f"{CODE}/05_openvocab_sft")
    return hf_repo_id or f"/outputs/work/{tag}/adapter"


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=EVAL_TIMEOUT_S)
def evaluate_run(dataset: str, run_name: str, model_id: str, adapter: str | None = None,
                 max_new_tokens: int = 6144, limit: int | None = None, subset_repo: str | None = None,
                 extra: list[str] | None = None) -> dict:
    outputs.reload()  # see adapters / partial generations written by other containers
    # MedGemma (no native thinking mode) is decoded greedily, like its earlier baseline runs;
    # Qwen3 in thinking mode must be sampled (its model card warns against greedy decoding)
    medgemma = "medgemma" in model_id.lower()
    cmd = ["python", "evaluate.py", "--model-type", "medgemma" if medgemma else "causal", "--engine", "vllm",
           "--model-id", model_id, "--run-name", run_name, "--dataset-name", dataset,
           "--output-dir", "/outputs/eval", "--max-new-tokens", str(max_new_tokens), "--batch-size", "100"]
    cmd += [] if medgemma else ["--do-sample"]
    cmd += ["--adapter", adapter] if adapter else []
    cmd += ["--limit", str(limit)] if limit else []
    cmd += ["--subset-repo-id", subset_repo] if subset_repo else []
    cmd += list(extra or [])
    _run(cmd, f"{CODE}/04_evaluation")
    metrics = json.loads(Path(f"/outputs/eval/{dataset}/{run_name}/metrics.json").read_text())
    return {"dataset": dataset, "run": run_name, "F1": metrics["F1_micro_type_agnostic"],
            "P": metrics["precision_type_agnostic"], "R": metrics["recall_type_agnostic"],
            "parse_errors": metrics["parse_status_counts"].get("parse_error", 0)}


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def pipeline(model_id: str, slug: str, variants: list[tuple[str, str]], eval_base: bool = True) -> list[dict]:
    """Trains every (tag, data_path) variant in parallel while the base
    model is evaluated, then evaluates each fine-tuned adapter as soon as
    its training ends. Adapters stay in the outputs volume
    (work/<slug>_<tag>/adapter). Runs on Modal, so the local client can
    disconnect (modal run --detach)."""
    trainings = {tag: train.spawn(model_id, data, f"{slug}_{tag}", None) for tag, data in variants}
    results = list(evaluate_run.starmap([(ds, f"{slug}_base", model_id, None, 6144, None,
                                          PILENER if ds == "pilener" else None) for ds in ALL])) if eval_base else []
    for tag, call in trainings.items():
        adapter = call.get()
        results += list(evaluate_run.starmap([(ds, f"{slug}_sft_{tag}", model_id, adapter, 6144, None,
                                                PILENER if ds == "pilener" else None) for ds in ALL]))
    for r in results:
        print(f"{r['run']:<28} {r['dataset']:<17} F1={r['F1']:.3f} P={r['P']:.2f} R={r['R']:.2f} "
              f"parse_errors={r['parse_errors']}")
    return results


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def evaluate_grid(model_id: str, runs: list[tuple[str, str | None]], max_new_tokens: int = 6144,
                  datasets: list[str] = DATASETS, subset_repo: str | None = None) -> list[dict]:
    """Evaluates every (run_name, adapter) on the 9 benchmarks, in parallel.
    adapter: None (base model), an HF repo, or a path in the outputs
    volume (e.g. an adapter trained elsewhere and uploaded with
    `modal volume put tesi-outputs <dir> work/<tag>/adapter`)."""
    # a run name ending in "_nothink" evaluates with Qwen3's thinking disabled; one ending in
    # "_desc" adds zero-shot entity-type definitions to the prompt (04_evaluation/type_descriptions_bio.json)
    def extra(run, ds):
        desc = "type_descriptions_crossner.json" if ds.startswith("crossner") else "type_descriptions_bio.json"
        return (["--no-thinking"] if run.endswith("_nothink") else
                ["--type-descriptions", desc] if run.endswith("_desc") else None)
    jobs = [(ds, run, model_id, adapter, max_new_tokens, None, subset_repo, extra(run, ds))
            for run, adapter in runs for ds in datasets]
    results = list(evaluate_run.starmap(jobs))
    for r in results:
        print(f"{r['run']:<28} {r['dataset']:<17} F1={r['F1']:.3f} P={r['P']:.2f} R={r['R']:.2f} "
              f"parse_errors={r['parse_errors']}")
    return results


# --- free vs structured CoT experiment (tutors' dataset and evaluation protocol) ---
TUTOR_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
TUTOR_EVAL = [f"tutor_eval/data/crossner_{d}.json" for d in ["ai", "literature", "music", "politics", "science"]] \
    + ["tutor_eval/data/pilener_test.json"]
# every training example is kept (longest rendered free example: 4,909 tokens), so that both
# students see the same 3,822 queries; other hyperparameters are train_qwen.py's defaults
TUTOR_TRAIN_ARGS = ["--max-seq-length", "5120"]


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=EVAL_TIMEOUT_S)
def evaluate_tutor(run_name: str, prompt: str, adapter: str | None = None, limit: int | None = None,
                   max_new_tokens: int = 12288, data: list[str] | None = None) -> None:
    """One model + prompt on the 5 CrossNER domains and the Pile-NER test split (one vLLM load)."""
    outputs.reload()
    cmd = ["python", "eval_tutor.py", "--model-id", TUTOR_MODEL, "--prompt", prompt, "--run-name", run_name,
           "--output-dir", "/outputs/eval_tutor", "--max-new-tokens", str(max_new_tokens),
           "--data", *(data or TUTOR_EVAL)]
    cmd += ["--adapter", adapter] if adapter else []
    cmd += ["--limit", str(limit)] if limit else []
    _run(cmd, f"{CODE}/04_evaluation")


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def tutor_pipeline(variants: list[str], train_extra: list[str], limit: int | None, prefix: str,
                   eval_base: bool = True) -> None:
    """Trains one adapter per CoT variant while the base model is evaluated with
    both prompts, then evaluates each adapter with the prompt it was trained on."""
    trainings = {v: train.spawn(TUTOR_MODEL, f"{CODE}/tutor_data/{v}.jsonl", f"{prefix}_{v}", None,
                                TUTOR_TRAIN_ARGS + train_extra) for v in variants}
    evals = [evaluate_tutor.spawn(f"{prefix}_base_{p}", p, None, limit) for p in variants] if eval_base else []
    for v, call in trainings.items():
        evals.append(evaluate_tutor.spawn(f"{prefix}_sft_{v}", v, call.get(), limit))
    for call in evals:
        call.get()


@app.local_entrypoint()
def tutor_smoke():
    """End-to-end check of the experiment, about 1 $: 3 training steps on 40
    structured examples, then 5 rows per eval set with the base model and
    that adapter (structured prompt)."""
    tutor_pipeline.remote(["structured"], ["--n-train-samples", "40", "--max-steps", "3", "--eval-steps", "3"],
                          5, "smoke")


@app.local_entrypoint()
def tutor_eval(run_name: str, prompt: str = "free", adapter: str = "", limit: int = 0, datasets: str = ""):
    """One evaluation only, e.g. the base model (no --adapter) on part of the sets:
    --run-name minirun_base_free --limit 50 --datasets crossner_ai"""
    data = [f"tutor_eval/data/{d}.json" for d in datasets.split(",")] if datasets else None
    evaluate_tutor.remote(run_name, prompt, adapter or None, limit or None, 12288, data)


@app.local_entrypoint()
def tutor():
    """The full experiment (run detached): free and structured students, base with both prompts."""
    tutor_pipeline.remote(["free", "structured"], [], None, "qwen3-4b-thinking")


@app.local_entrypoint()
def grid(model_id: str = "Qwen/Qwen3-1.7B", runs: str = "base", datasets: str = "", subset_repo: str = ""):
    """e.g. --runs base,v3=/outputs/work/qwen3-1.7b_v3/adapter  (run names get the model slug prefix);
    --datasets pilener --subset-repo filippogreppi/pilener-eval-subset for the in-domain Pile-NER set"""
    slug = model_id.split("/")[-1].lower()
    parsed = []
    for item in runs.split(","):
        name, _, adapter = item.partition("=")
        parsed.append((f"{slug}_{name}" if name.startswith("base") else f"{slug}_sft_{name}", adapter or None))
    ds_list = datasets.split(",") if datasets else DATASETS
    for r in evaluate_grid.remote(model_id, parsed, datasets=ds_list, subset_repo=subset_repo or None):
        print(r)


@app.local_entrypoint()
def smoke():
    """End-to-end check for a few cents: 3 training steps on Qwen3-1.7B,
    then 10 rows of ncbi with the base model and with that adapter."""
    adapter = train.remote("Qwen/Qwen3-1.7B", HF_DATA, "smoke", None,
                           ["--n-train-samples", "24", "--max-steps", "3", "--eval-steps", "3"])
    for a, run in [(None, "smoke_base"), (adapter, "smoke_sft")]:
        print(evaluate_run.remote("ncbi", run, "Qwen/Qwen3-1.7B", a, max_new_tokens=2048, limit=10))


@app.local_entrypoint()
def smoke_eval(adapter: str = "", model_id: str = "Qwen/Qwen3-1.7B", run_name: str = ""):
    """Only the vLLM evaluation path, 5 rows of ncbi."""
    print(evaluate_run.remote("ncbi", run_name or ("smoke_eval_sft" if adapter else "smoke_eval_base"), model_id,
                              adapter or None, max_new_tokens=2048, limit=5))


@app.local_entrypoint()
def full(model_id: str = "Qwen/Qwen3-4B",
         variants: str = "v4=openvocab_train_v4_qwen.jsonl,v4lo=openvocab_train_v4_labelonly.jsonl",
         eval_base: bool = True):
    """variants: tag=<file in the filippogreppi/openvocab-cot-sft dataset repo>, comma-separated"""
    slug = model_id.split("/")[-1].lower()
    parsed = [(tag, f"hf://filippogreppi/openvocab-cot-sft/{f}") for tag, f in (v.split("=") for v in variants.split(","))]
    for r in pipeline.remote(model_id, slug, parsed, eval_base):
        print(r)
