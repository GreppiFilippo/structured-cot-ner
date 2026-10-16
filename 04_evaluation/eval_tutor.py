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
import time
from collections import defaultdict
from pathlib import Path

from src.metrics import _loads_or_none, exact_counts, f1_from_counts, iter_json_objects

THINK_END = "</think>"

# Grammar of the tutors' structured run (dataset card: thinking constrained to three one-line
# fields, then the JSON answer constrained to a generic {type: [strings]} shape), in xgrammar's
# EBNF. The generation prompt already ends with "<think>\n", so the output starts at TYPES.
_FIELDS = 'root ::= "TYPES: " line "MENTIONS: " line "EDGE: " line "</think>\\n\\n" answer\n' \
          'line ::= [^\\n]+ "\\n"\n'
_JSON = ('list ::= "[" (str ("," str)*)? "]"\n'
         'str ::= "\\"" ([^"\\\\\\n] | "\\\\" ["\\\\/bfnrt] | "\\\\u" hex hex hex hex)* "\\""\n'
         'hex ::= [0-9a-fA-F]\n')


def grammar_for(mode: str, requested: list[str]) -> str | None:
    """none: free decoding; tutors: the tutors' grammar (any keys); keys: the same thinking
    fields, with the answer's keys fixed to the requested types, in their order."""
    if mode == "none":
        return None
    if mode == "tutors":
        return _FIELDS + 'answer ::= "{" (pair ("," pair)*)? "}"\npair ::= str ":" list\n' + _JSON
    keys = ' "," '.join(f'"\\"{t}\\":" list' for t in requested)
    return _FIELDS + f'answer ::= "{{" {keys} "}}"\n' + _JSON


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


def budget_forced(llm, tok, prompts, rows, params, lora, think_end_id, budget_file):
    """Two passes (budget forcing, as in s1): the reasoning is generated up to the query's
    budget; if it has not closed by then, "\n</think>\n\n" is appended and the answer is
    generated from there. Returns (token ids, text, finish reason, forced) per query."""
    budget = {json.loads(l)["id"]: json.loads(l)["reasoning_tokens"]
              for l in budget_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    first = []
    for r in rows:
        p = params.clone()
        p.max_tokens = max(1, budget[r["id"]])
        first.append(p)
    outs = llm.generate(prompts, first, lora_request=lora)
    close = tok("\n</think>\n\n", add_special_tokens=False)["input_ids"]
    result, todo = [None] * len(rows), []
    for i, o in enumerate(outs):
        g = o.outputs[0]
        ids = list(g.token_ids)
        if think_end_id in ids or g.finish_reason != "length":
            # closed within budget: a reply already started after </think> is finished like the rest
            if think_end_id in ids and g.finish_reason == "length":
                todo.append((i, ids, False))
            else:
                result[i] = (ids, g.text, g.finish_reason, False)
        else:
            todo.append((i, ids + close, True))
    if todo:
        second = []
        for i, ids, _ in todo:
            p = params.clone()
            p.max_tokens = max(1, params.max_tokens - len(ids))
            second.append(p)
        outs2 = llm.generate([{"prompt_token_ids": prompts[i]["prompt_token_ids"] + ids} for i, ids, _ in todo],
                             second, lora_request=lora)
        for (i, ids, forced), o in zip(todo, outs2):
            g = o.outputs[0]
            full = ids + list(g.token_ids)
            result[i] = (full, tok.decode(full, skip_special_tokens=False), g.finish_reason, forced)
    return result


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
    # on a 24 GB RTX 3090, 0.9 / 256 sequences ran out of memory in the top-k/top-p sampler, which
    # sorts the full-vocabulary logits of every running sequence outside vLLM's reserved memory
    ap.add_argument("--grammar", choices=["none", "tutors", "keys"], default="none",
                    help="Constrained decoding (see grammar_for); default: free decoding")
    ap.add_argument("--think-budget-from", default=None, metavar="RUN",
                    help="Budget forcing: cap each query's reasoning at the tokens run RUN (same --output-dir) "
                         "used on it; a reasoning still open there is closed with </think> and the answer "
                         "is generated after it")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    args = ap.parse_args()

    import transformers
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

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
              max_model_len=longest + args.max_new_tokens, gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs)
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20,
                            seed=args.seed, skip_special_tokens=False)

    for name, rows in data.items():
        prompts = [{"prompt_token_ids": tok(t, add_special_tokens=False)["input_ids"]} for t in texts[name]]
        per_request = params
        if args.grammar != "none":
            per_request = [params.clone() for _ in rows]
            for p, r in zip(per_request, rows):
                p.guided_decoding = GuidedDecodingParams(grammar=grammar_for(args.grammar, r["requested_types"]),
                                                         backend="xgrammar")
        t0 = time.time()
        if args.think_budget_from:
            gens = budget_forced(llm, tok, prompts, rows, params, lora, think_end_id,
                                 Path(args.output_dir) / name / args.think_budget_from / "predictions.jsonl")
        else:
            gens = [(list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].finish_reason, False)
                    for o in llm.generate(prompts, per_request, lora_request=lora)]
        elapsed = time.time() - t0
        records = []
        for r, (ids, text, finish, forced) in zip(rows, gens):
            n_reason = ids.index(think_end_id) if think_end_id in ids else len(ids)
            pred, status = parse_answer(text, r["requested_types"])
            records.append({"id": r["id"], "requested_types": r["requested_types"], "distractors": r["distractors"],
                            "gold": r["gold"], "pred": pred, "status": status, "finish_reason": finish,
                            "reasoning_tokens": n_reason, "output_tokens": len(ids), "budget_forced": forced,
                            "raw": text})
        out = Path(args.output_dir) / name / args.run_name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        m = score(records) | {"run": args.run_name, "dataset": name, "model_id": args.model_id,
                              "adapter": args.adapter, "prompt": args.prompt, "grammar": args.grammar,
                              "think_budget_from": args.think_budget_from,
                              "budget_forced": sum(r["budget_forced"] for r in records),
                              "seed": args.seed,
                              "max_new_tokens": args.max_new_tokens, "generation_seconds": elapsed,
                              "output_tokens_per_second": sum(r["output_tokens"] for r in records) / elapsed,
                              "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}}
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"{args.run_name:<22} {name:<20} F1={m['F1']:.3f} (no MISC {m['without_MISC']['F1']:.3f}) "
              f"P={m['P']:.3f} R={m['R']:.3f} reasoning={m['mean_reasoning_tokens']:.0f} tok "
              f"truncated={m['truncated']} status={m['status_counts']} "
              f"[{elapsed / 60:.1f} min, {m['output_tokens_per_second']:.0f} tok/s]", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
