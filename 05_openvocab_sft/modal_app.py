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
# the same cap for a training container (TRAIN_TIMEOUT_S=18000 modal run ...)
TRAIN_TIMEOUT_S = int(os.environ.get("TRAIN_TIMEOUT_S", 24 * 3600))
GPU = "L40S"  # 48 GB, faster than an RTX 3090; ~2 $/h

DATASETS = ["ncbi", "bc5cdr", "bc2gm", "bc4chemd", "AnatEM", "JNLPBA", "JNLPBA-Rare", "MedMentions-Rare", "biored"]
HF_DATA = "hf://structured-cot-ner/openvocab-sft/openvocab_train_v3_qwen.jsonl"
# the 9 benchmarks + the in-domain Pile-NER test (its own subset repo)
ALL = DATASETS + ["pilener"]
PILENER = "structured-cot-ner/pilener-eval-subset"

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
# training files of the free vs structured CoT experiment (build_paired_cot_dataset.py; not in git,
# derived from the private paired-trace dataset): added only where they have been built
# (and their MedGemma versions, make_medgemma_paired_data.py, and the structure without EDGE,
# make_noedge_data.py)
for _v in ("free", "structured", "noedge"):
    for _src, _dst in ((f"{_v}_cot", _v), (f"{_v}_cot_medgemma", f"{_v}_medgemma")):
        _f = REPO / f"05_openvocab_sft/data/pilener_paired_{_src}.jsonl"
        if _f.exists():
            image = image.add_local_file(_f, f"{CODE}/paired_data/{_dst}.jsonl")
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


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=TRAIN_TIMEOUT_S)
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


# --- free vs structured CoT experiment (paired-trace dataset and its evaluation protocol) ---
PAIRED_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
PAIRED_EVAL = [f"paired_eval/data/crossner_{d}.json" for d in ["ai", "literature", "music", "politics", "science"]] \
    + ["paired_eval/data/pilener_test.json"]
# every training example is kept (longest rendered free example: 4,909 tokens), so that both
# students see the same 3,822 queries; other hyperparameters are train_qwen.py's defaults
PAIRED_TRAIN_ARGS = ["--max-seq-length", "5120"]


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=EVAL_TIMEOUT_S)
def evaluate_paired(run_name: str, prompt: str, adapter: str | None = None, limit: int | None = None,
                   max_new_tokens: int = 12288, data: list[str] | None = None) -> None:
    """One model + prompt on the 5 CrossNER domains and the Pile-NER test split (one vLLM load)."""
    outputs.reload()
    cmd = ["python", "eval_paired.py", "--model-id", PAIRED_MODEL, "--prompt", prompt, "--run-name", run_name,
           "--output-dir", "/outputs/eval_paired", "--max-new-tokens", str(max_new_tokens),
           "--data", *(data or PAIRED_EVAL)]
    cmd += ["--adapter", adapter] if adapter else []
    cmd += ["--limit", str(limit)] if limit else []
    _run(cmd, f"{CODE}/04_evaluation")


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def paired_pipeline(variants: list[str], train_extra: list[str], limit: int | None, prefix: str,
                   eval_base: bool = True) -> None:
    """Trains one adapter per CoT variant while the base model is evaluated with
    both prompts, then evaluates each adapter with the prompt it was trained on."""
    trainings = {v: train.spawn(PAIRED_MODEL, f"{CODE}/paired_data/{v}.jsonl", f"{prefix}_{v}", None,
                                PAIRED_TRAIN_ARGS + train_extra) for v in variants}
    evals = [evaluate_paired.spawn(f"{prefix}_base_{p}", p, None, limit) for p in variants] if eval_base else []
    for v, call in trainings.items():
        evals.append(evaluate_paired.spawn(f"{prefix}_sft_{v}", v, call.get(), limit))
    for call in evals:
        call.get()


@app.local_entrypoint()
def paired_smoke():
    """End-to-end check of the experiment, about 1 $: 3 training steps on 40
    structured examples, then 5 rows per eval set with the base model and
    that adapter (structured prompt)."""
    paired_pipeline.remote(["structured"], ["--n-train-samples", "40", "--max-steps", "3", "--eval-steps", "3"],
                          5, "smoke")


@app.local_entrypoint()
def paired_eval(run_name: str, prompt: str = "free", adapter: str = "", limit: int = 0, datasets: str = ""):
    """One evaluation only, e.g. the base model (no --adapter) on part of the sets:
    --run-name minirun_base_free --limit 50 --datasets crossner_ai"""
    data = [f"paired_eval/data/{d}.json" for d in datasets.split(",")] if datasets else None
    evaluate_paired.remote(run_name, prompt, adapter or None, limit or None, 12288, data)


@app.local_entrypoint()
def paired():
    """The full experiment (run detached): free and structured students, base with both prompts."""
    paired_pipeline.remote(["free", "structured"], [], None, "qwen3-4b-thinking")


# Same experiment on a student of another family. SmolLM3-3B reasons in a <think>
# block that the model opens itself (its template does not add it to the generation
# prompt), so the training files of the Qwen students are used unchanged.
SMOL_MODEL = "HuggingFaceTB/SmolLM3-3B"
PAIRED_BIO = [f"paired_eval/data_bio/{d}.json" for d in DATASETS]
# evaluation sets per container, with their output directory
PAIRED_GROUPS = [(PAIRED_EVAL[:5], "/outputs/eval_paired"), (PAIRED_EVAL[5:], "/outputs/eval_paired"),
                 (PAIRED_BIO, "/outputs/eval_paired_bio")]


@app.function(gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=EVAL_TIMEOUT_S)
def evaluate_family(model_id: str, run_name: str, prompt: str, adapter: str | None, limit: int | None,
                    data: list[str], output_dir: str) -> None:
    """One model + prompt on a list of evaluation sets (one vLLM load)."""
    outputs.reload()
    cmd = ["python", "eval_paired.py", "--model-id", model_id, "--prompt", prompt, "--run-name", run_name,
           "--output-dir", output_dir, "--max-new-tokens", "12288", "--data", *data]
    cmd += ["--adapter", adapter] if adapter else []
    cmd += ["--limit", str(limit)] if limit else []
    _run(cmd, f"{CODE}/04_evaluation")


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def family_pipeline(model_id: str, prefix: str, train_extra: list[str], limit: int | None,
                    data_suffix: str = "", eval_base: bool = False,
                    variants: tuple[str, ...] = ("free", "structured"),
                    groups: list = PAIRED_GROUPS, base_groups: list | None = None) -> None:
    """Trains a student of model_id per CoT variant (by default free and structured) in
    parallel, then evaluates each with its own prompt on CrossNER, on the Pile-NER test
    split and on the nine biomedical sets, one container per group of sets (a stopped
    container loses only its own group). eval_base also evaluates the base model with
    the free prompt, while the students train (base_groups: its own split, default groups)."""
    trainings = {v: train.spawn(model_id, f"{CODE}/paired_data/{v}{data_suffix}.jsonl", f"{prefix}_{v}", None,
                                PAIRED_TRAIN_ARGS + train_extra) for v in variants}
    evals = [evaluate_family.spawn(model_id, f"{prefix}_base_free", "free", None, limit, data, out)
             for data, out in (base_groups or groups)] if eval_base else []
    for v, call in trainings.items():
        adapter = call.get()
        evals += [evaluate_family.spawn(model_id, f"{prefix}_sft_{v}", v, adapter, limit, data, out)
                  for data, out in groups]
    for call in evals:
        call.get()


@app.local_entrypoint()
def noedge_smoke():
    """End-to-end check of the structure without EDGE: 3 training steps, 5 rows per eval set."""
    family_pipeline.remote(PAIRED_MODEL, "smoke_qwen3-4b-thinking",
                           ["--n-train-samples", "40", "--max-steps", "3", "--eval-steps", "3"], 5,
                           variants=("noedge",))


@app.local_entrypoint()
def noedge():
    """Qwen3-4B-Thinking-2507 student on the structured traces without EDGE (run detached)."""
    family_pipeline.remote(PAIRED_MODEL, "qwen3-4b-thinking", [], None, variants=("noedge",))


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def evaluate_trained(model_id: str, run_name: str, prompt: str, adapter: str) -> None:
    """The three evaluation groups of family_pipeline for an adapter already on the volume."""
    groups = [(PAIRED_EVAL[:5], "/outputs/eval_paired"), (PAIRED_EVAL[5:], "/outputs/eval_paired"),
              (PAIRED_BIO, "/outputs/eval_paired_bio")]
    for call in [evaluate_family.spawn(model_id, run_name, prompt, adapter, None, data, out)
                 for data, out in groups]:
        call.get()


@app.local_entrypoint()
def noedge_eval():
    """Evaluations of the no-EDGE student whose adapter is already trained (run detached,
    keeping the client attached: a client stopped early cancels the remote call)."""
    evaluate_trained.remote(PAIRED_MODEL, "qwen3-4b-thinking_sft_noedge", "noedge",
                            "/outputs/work/qwen3-4b-thinking_noedge/adapter")


@app.local_entrypoint()
def smol_smoke():
    """End-to-end check with SmolLM3-3B: 3 training steps per variant, 5 rows per eval set."""
    family_pipeline.remote(SMOL_MODEL, "smoke_smollm3",
                           ["--n-train-samples", "40", "--max-steps", "3", "--eval-steps", "3"], 5)


@app.local_entrypoint()
def smol():
    """Free and structured SmolLM3-3B students (run detached)."""
    family_pipeline.remote(SMOL_MODEL, "smollm3-3b", [], None)


# Qwen3-8B, same protocol as Qwen3-1.7B and Qwen3-0.6B (sbatch_paired_small.sh). In bf16 it
# does not fit in the 24 GB of an RTX 3090 on the longest free examples, so it runs here.
# The sets are split over more containers, to use the 10 GPUs of the account at once:
# 2 trainings + 8 base groups, then 4 groups per student.
QWEN8B_MODEL = "Qwen/Qwen3-8B"
_CROSSNER, _PILENER = PAIRED_EVAL[:5], PAIRED_EVAL[5:]
QWEN8B_BASE_GROUPS = [(_CROSSNER[:2], "/outputs/eval_paired"), (_CROSSNER[2:4], "/outputs/eval_paired"),
                      (_CROSSNER[4:], "/outputs/eval_paired"), (_PILENER, "/outputs/eval_paired")] \
    + [(PAIRED_BIO[i:i + 2], "/outputs/eval_paired_bio") for i in (0, 2, 4)] \
    + [(PAIRED_BIO[6:], "/outputs/eval_paired_bio")]
QWEN8B_GROUPS = [(_CROSSNER, "/outputs/eval_paired"), (_PILENER, "/outputs/eval_paired"),
                 (PAIRED_BIO[:5], "/outputs/eval_paired_bio"), (PAIRED_BIO[5:], "/outputs/eval_paired_bio")]


@app.local_entrypoint()
def qwen8b_smoke():
    """Speed check before the full run (about 1 $): 3 training steps on the free traces,
    then 20 rows of CrossNER AI and of the Pile-NER test with the base model and that adapter."""
    groups = [(_CROSSNER[:1] + _PILENER, "/outputs/eval_paired")]
    family_pipeline.remote(QWEN8B_MODEL, "smoke_qwen3-8b",
                           ["--n-train-samples", "40", "--max-steps", "3", "--eval-steps", "3"], 20,
                           eval_base=True, variants=("free",), groups=groups)


@app.local_entrypoint()
def qwen8b():
    """Free and structured Qwen3-8B students and the base model (run detached, with
    TRAIN_TIMEOUT_S and EVAL_TIMEOUT_S set)."""
    family_pipeline.remote(QWEN8B_MODEL, "qwen3-8b", [], None, eval_base=True,
                           groups=QWEN8B_GROUPS, base_groups=QWEN8B_BASE_GROUPS)


# MedGemma-1.5-4B, the medical student of the first phase: same queries, prompts,
# reasoning and answers, written in its own thought block (make_medgemma_paired_data.py);
# eval_paired.py reads that block (--thought-format auto).
MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def catch_up_pipeline() -> None:
    """Evaluations queued on the cluster, moved here for speed (adapters uploaded to
    /outputs/work/<tag>/adapter with `modal volume put`): the biomedical sets for the
    follow-up Qwen3-4B-Thinking-2507 students, and the Pile-NER test split for the
    SmolLM3 students. One container per evaluation, all in parallel; datasets already
    evaluated are skipped by eval_paired.py."""
    jobs = [(PAIRED_MODEL, f"qwen3-4b-thinking_sft_{t}", "structured" if t.startswith("structured") else "free",
             f"/outputs/work/qwen3-4b-thinking_{t}/adapter", None, PAIRED_BIO, "/outputs/eval_paired_bio")
            for t in ["free", "structured", "noreason", "freetrunc", "free_seed1", "structured_seed1",
                      "free_seed2", "structured_seed2"]]
    jobs += [(SMOL_MODEL, f"smollm3-3b_sft_{v}", v, f"/outputs/work/smollm3-3b_{v}/adapter", None,
              PAIRED_EVAL[5:], "/outputs/eval_paired") for v in ("free", "structured")]
    list(evaluate_family.starmap(jobs))


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def base_pipeline(model_id: str, run_name: str, bio_sets: list[str] | None = None) -> None:
    """A base model with the free prompt on CrossNER, on the Pile-NER test split and on the
    biomedical sets (all nine, or bio_sets), one container per group, in parallel."""
    bio = [f"paired_eval/data_bio/{d}.json" for d in bio_sets] if bio_sets else PAIRED_BIO
    groups = [(bio, "/outputs/eval_paired_bio")] if bio_sets else \
        [(PAIRED_EVAL[:5], "/outputs/eval_paired"), (PAIRED_EVAL[5:], "/outputs/eval_paired"), (bio, "/outputs/eval_paired_bio")]
    list(evaluate_family.starmap([(model_id, run_name, "free", None, None, data, out) for data, out in groups]))


@app.local_entrypoint()
def smol_base():
    """Base SmolLM3-3B with the free prompt (run detached)."""
    base_pipeline.remote(SMOL_MODEL, "smollm3-3b_base_free")


@app.function(volumes=VOLUMES, timeout=24 * 3600)
def base_bio_sets(model_id: str, run_name: str, sets: list[str]) -> None:
    """A base model with the free prompt on some biomedical sets, one container per set."""
    list(evaluate_family.starmap([(model_id, run_name, "free", None, None, [f"paired_eval/data_bio/{d}.json"],
                                   "/outputs/eval_paired_bio") for d in sets]))


@app.local_entrypoint()
def qwen_base_bio(sets: str = "JNLPBA-Rare,MedMentions-Rare"):
    """Base Qwen3-4B-Thinking-2507 with the free prompt on biomedical sets not yet evaluated (comma-separated)."""
    base_bio_sets.remote(PAIRED_MODEL, "qwen3-4b-thinking_base_free", sets.split(","))


@app.local_entrypoint()
def catch_up():
    """Runs catch_up_pipeline remotely, so that a detached run survives the local
    process (a starmap called from a local entrypoint is cancelled with it)."""
    catch_up_pipeline.remote()


@app.local_entrypoint()
def medgemma_smoke():
    """End-to-end check with MedGemma: 3 training steps per variant, base and students on 5 rows per set."""
    family_pipeline.remote(MEDGEMMA_MODEL, "smoke_medgemma",
                           ["--n-train-samples", "40", "--max-steps", "3", "--eval-steps", "3"], 5,
                           "_medgemma", True)


@app.local_entrypoint()
def medgemma():
    """Free and structured MedGemma students, and the base model (run detached)."""
    family_pipeline.remote(MEDGEMMA_MODEL, "medgemma-1.5-4b", [], None, "_medgemma", True)


@app.local_entrypoint()
def grid(model_id: str = "Qwen/Qwen3-1.7B", runs: str = "base", datasets: str = "", subset_repo: str = ""):
    """e.g. --runs base,v3=/outputs/work/qwen3-1.7b_v3/adapter  (run names get the model slug prefix);
    --datasets pilener --subset-repo structured-cot-ner/pilener-eval-subset for the in-domain Pile-NER set"""
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
    """variants: tag=<file in the structured-cot-ner/openvocab-sft dataset repo>, comma-separated"""
    slug = model_id.split("/")[-1].lower()
    parsed = [(tag, f"hf://structured-cot-ner/openvocab-sft/{f}") for tag, f in (v.split("=") for v in variants.split(","))]
    for r in pipeline.remote(model_id, slug, parsed, eval_base):
        print(r)
