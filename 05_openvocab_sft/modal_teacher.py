"""Schema-guided teacher generation on Modal (vLLM).

The first teacher run produced open-schema CoTs (no entity types given), so
its reasoning often disagrees with the requested types and the Pile-NER gold
the SFT examples pair it with. Here the teacher (Qwen3-32B, thinking mode)
receives EXACTLY the prompt the student sees — system + user with the list of
ENTITY TYPES — as in ReasoningNER, where the CoT is generated conditioned on
the schema. Reasoning and answer then come from the same call.

Input: an open-vocabulary .jsonl ({"id", "messages", ...}); only the system
and user turns are used. Output (in the outputs volume): teacher/<tag>.jsonl,
one row per example: id, reasoning, answer (text after </think>),
finish_reason, n_tokens. Resumable: rows already written are skipped.

    # 20 rows, to measure speed and cost
    .venv/bin/modal run 05_openvocab_sft/modal_teacher.py::run --tag smoke --limit 20
    # all rows, detached
    .venv/bin/modal run --detach 05_openvocab_sft/modal_teacher.py::run --tag v4all_schema
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

TEACHER = "Qwen/Qwen3-32B-FP8"
SOURCE = "hf://filippogreppi/openvocab-cot-sft/openvocab_train_v4_all.jsonl"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm==0.30.0", "transformers==5.17.0", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)
hf_cache = modal.Volume.from_name("tesi-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("tesi-outputs", create_if_missing=True)
app = modal.App("tesi-teacher", image=image)


def _load_rows(source: str) -> list[dict]:
    import os
    from huggingface_hub import hf_hub_download
    owner, repo, filename = source[len("hf://"):].split("/", 2)
    path = hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _split(text: str) -> tuple[str, str]:
    """(reasoning, answer) from a thinking-mode completion."""
    if "</think>" not in text:
        return text.replace("<think>", "").strip(), ""
    head, answer = text.rsplit("</think>", 1)
    return head.replace("<think>", "").strip(), answer.strip()


@app.function(gpu="H100", volumes={"/cache": hf_cache, "/outputs": outputs},
              secrets=[modal.Secret.from_name("huggingface")], timeout=12 * 3600)
def generate(tag: str, source: str = SOURCE, limit: int | None = None, max_tokens: int = 6144,
             chunk: int = 512) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = _load_rows(source)[:limit] if limit else _load_rows(source)
    out_path = Path(f"/outputs/teacher/{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already done, {len(todo)} to generate")

    tok = AutoTokenizer.from_pretrained(TEACHER)
    llm = LLM(TEACHER, max_model_len=10240, gpu_memory_utilization=0.92, seed=42)
    # Qwen3 thinking-mode sampling recommendations
    params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=42)

    t0, n_tok = time.time(), 0
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True) for r in batch]
        results = llm.generate(prompts, params)
        with out_path.open("a", encoding="utf-8") as f:
            for r, res in zip(batch, results):
                o = res.outputs[0]
                reasoning, answer = _split(o.text)
                n_tok += len(o.token_ids)
                f.write(json.dumps({"id": r["id"], "reasoning": reasoning, "answer": answer,
                                    "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids)},
                                   ensure_ascii=False) + "\n")
        outputs.commit()
        el = time.time() - t0
        print(f"{start + len(batch)}/{len(todo)} done, {n_tok} tokens, {el / 60:.1f} min, {n_tok / el:.0f} tok/s")
    hf_cache.commit()
    return {"rows": len(todo), "tokens": n_tok, "minutes": round((time.time() - t0) / 60, 1)}


@app.local_entrypoint()
def run(tag: str = "smoke", limit: int = 0, source: str = SOURCE):
    print(generate.remote(tag, source, limit or None))
