"""Evaluation of the free vs structured CoT experiment, with the tutors'
protocol, on the sets built by build_tutor_eval.py (CrossNER with gold types
+ 1-3 distractors, and the in-domain Pile-NER test split).

- Prompt: the user message only (no system prompt), "free" or "structured"
  (the prompt each model was trained on; the base model is run with both),
  rendered with the model's chat template. For Qwen3-4B-Thinking-2507 the
  generation prompt already opens the thinking block ("<think>\\n").
- Decoding: vLLM, sampling as recommended for Qwen3 thinking models
  (T 0.6, top-p 0.95, top-k 20), fixed seed.
- Answer: the JSON object after </think>. No </think> (generation cut at the
  token budget) or no parsable object = empty prediction, counted as such.
- Metric: micro F1 with exact, case-insensitive mention matching per type,
  over the requested types only (src.metrics.exact_counts); also without
  MISC, which the per-type counts make possible without new generations.
- Reasoning length: tokens generated before </think>.

Output, per dataset: <output-dir>/<dataset>/<run-name>/{predictions.jsonl,metrics.json}

Usage (on a GPU machine, see 05_openvocab_sft/modal_app.py::tutor):
    python3 eval_tutor.py --model-id Qwen/Qwen3-4B-Thinking-2507 --prompt free \\
        --run-name base_free --data tutor_eval/data/crossner_ai.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from src.metrics import _loads_or_none, exact_counts, f1_from_counts, iter_json_objects

THINK_END = "</think>"


def parse_answer(text: str, requested: list[str]) -> tuple[dict, str]:
    """(prediction restricted to the requested types, status)."""
    if THINK_END not in text:
        return {}, "no_think_end"
    answer = text.split(THINK_END, 1)[1]
    obj = None
    for cand in iter_json_objects(answer):
        obj = _loads_or_none(cand)
        if isinstance(obj, dict):
            break
    if not isinstance(obj, dict):
        return {}, "parse_error"
    pred = {}
    for t in requested:
        v = obj.get(t, [])
        pred[t] = [m for m in (v if isinstance(v, list) else [v]) if isinstance(m, str)]
    return pred, "ok" if set(obj) == set(requested) else "ok_keys_differ"


def prf(c):
    tp, gt, pc = c
    return {"F1": f1_from_counts(tp, gt, pc), "P": tp / pc if pc else 0.0, "R": tp / gt if gt else 0.0,
            "TP": tp, "GT": gt, "PRED": pc}


def score(records: list[dict]) -> dict:
    total, no_misc = [0, 0, 0], [0, 0, 0]
    per_type = defaultdict(lambda: [0, 0, 0])
    for r in records:
        for t in r["requested_types"]:
            c = exact_counts({t: r["gold"][t]}, {t: r["pred"].get(t, [])})
            for acc in (total, per_type[t]) + ((no_misc,) if t != "MISC" else ()):
                for i in range(3):
                    acc[i] += c[i]
    reasoning = [r["reasoning_tokens"] for r in records]
    closed = [r["reasoning_tokens"] for r in records if r["status"] != "no_think_end"]
    status = defaultdict(int)
    for r in records:
        status[r["status"]] += 1
    return {
        "n": len(records), **prf(total), "without_MISC": prf(no_misc),
        "mean_reasoning_tokens": statistics.mean(reasoning),
        "median_reasoning_tokens": statistics.median(reasoning),
        "mean_reasoning_tokens_completed": statistics.mean(closed) if closed else None,
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in records),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
        "status_counts": dict(status),
        "per_type": {t: prf(c) for t, c in sorted(per_type.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--adapter", default=None, help="Local LoRA adapter directory")
    ap.add_argument("--prompt", choices=["free", "structured"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="Files written by build_tutor_eval.py")
    ap.add_argument("--output-dir", default="outputs_tutor")
    ap.add_argument("--max-new-tokens", type=int, default=12288, help="Default: the teacher's free-run budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="First N rows of each file (smoke tests)")
    args = ap.parse_args()

    import transformers
    from vllm import LLM, SamplingParams

    tok = transformers.AutoTokenizer.from_pretrained(args.model_id)
    think_end_id = tok.convert_tokens_to_ids(THINK_END)
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        from evaluate import _check_adapter_files
        _check_adapter_files(args.adapter)  # refuses an adapter whose LoRA B matrices are all zero
        lora = LoRARequest("adapter", 1, args.adapter)
    # a resubmitted job skips the datasets it already finished
    todo = [f for f in args.data if not (Path(args.output_dir) / Path(f).stem / args.run_name / "metrics.json").exists()]
    if not todo:
        print(f"{args.run_name}: every dataset already evaluated")
        return
    data = {Path(f).stem: json.loads(Path(f).read_text(encoding="utf-8"))[: args.limit] for f in todo}
    texts = {name: [tok.apply_chat_template([{"role": "user", "content": r[f"{args.prompt}_prompt"]}],
                                            add_generation_prompt=True, tokenize=False) for r in rows]
             for name, rows in data.items()}
    longest = max(len(tok(t, add_special_tokens=False)["input_ids"]) for ts in texts.values() for t in ts)
    llm = LLM(model=args.model_id, dtype="bfloat16", seed=args.seed, enable_lora=bool(lora), max_lora_rank=16,
              max_model_len=longest + args.max_new_tokens, gpu_memory_utilization=0.9)
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20,
                            seed=args.seed, skip_special_tokens=False)

    for name, rows in data.items():
        prompts = [{"prompt_token_ids": tok(t, add_special_tokens=False)["input_ids"]} for t in texts[name]]
        outs = llm.generate(prompts, params, lora_request=lora)
        records = []
        for r, o in zip(rows, outs):
            gen = o.outputs[0]
            ids = list(gen.token_ids)
            n_reason = ids.index(think_end_id) if think_end_id in ids else len(ids)
            pred, status = parse_answer(gen.text, r["requested_types"])
            records.append({"id": r["id"], "requested_types": r["requested_types"], "distractors": r["distractors"],
                            "gold": r["gold"], "pred": pred, "status": status, "finish_reason": gen.finish_reason,
                            "reasoning_tokens": n_reason, "output_tokens": len(ids), "raw": gen.text})
        out = Path(args.output_dir) / name / args.run_name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        m = score(records) | {"run": args.run_name, "dataset": name, "model_id": args.model_id,
                              "adapter": args.adapter, "prompt": args.prompt, "seed": args.seed,
                              "max_new_tokens": args.max_new_tokens,
                              "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}}
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"{args.run_name:<22} {name:<20} F1={m['F1']:.3f} (no MISC {m['without_MISC']['F1']:.3f}) "
              f"P={m['P']:.3f} R={m['R']:.3f} reasoning={m['mean_reasoning_tokens']:.0f} tok "
              f"truncated={m['truncated']} status={m['status_counts']}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
