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
import subprocess
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
CODE = "/root/code"
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


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=8 * 3600)
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
    # a run name ending in "_nothink" evaluates with Qwen3's thinking disabled
    jobs = [(ds, run, model_id, adapter, max_new_tokens, None, subset_repo,
             ["--no-thinking"] if run.endswith("_nothink") else None)
            for run, adapter in runs for ds in datasets]
    results = list(evaluate_run.starmap(jobs))
    for r in results:
        print(f"{r['run']:<28} {r['dataset']:<17} F1={r['F1']:.3f} P={r['P']:.2f} R={r['R']:.2f} "
              f"parse_errors={r['parse_errors']}")
    return results


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
