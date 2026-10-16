"""Published open NER models evaluated with the protocol of the free vs
structured CoT experiment: the same queries (build_tutor_eval.py /
build_tutor_eval_bio.py: gold types + 1-3 in-domain distractors), the same
exact typed F1 (eval_tutor.score), the same output files, so they are
compared with the students by compare_tutor.py-style scripts.

- gliner: GLiNER-L v2.1 (urchade/gliner_large-v2.1, trained on Pile text
  annotated by Mistral), predict_entities with its default threshold 0.5.
- uniner: UniNER-7B-type (Universal-NER/UniNER-7B-type), with the prompt of
  its model card, one question per requested type, greedy decoding, 256 new
  tokens, as in the authors' inference script.

Both models take entity types in natural language, so every type name is
given lower-cased with spaces (MUSICAL_ARTIST -> "musical artist") and the
predictions are mapped back. Neither generates a reasoning: reasoning_tokens
is 0. Pile-NER is not used for them: both were trained on Pile passages.

Usage:
    python3 eval_baselines.py --model gliner --run-name gliner-l-v2.1 \\
        --data tutor_eval/data/crossner_ai.json --output-dir outputs_tutor
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval_tutor import score
from src.metrics import _loads_or_none, iter_json_objects

GLINER = "urchade/gliner_large-v2.1"
UNINER = "Universal-NER/UniNER-7B-type"
UNINER_PROMPT = ("A virtual assistant answers questions from a user based on the provided text. "
                 "USER: Text: {text} ASSISTANT: I’ve read this text.</s>"
                 "USER: What describes {etype} in the text? ASSISTANT:")


def natural(etype: str) -> str:
    return etype.lower().replace("_", " ")


def run_gliner(rows: list[dict], threshold: float) -> list[tuple[dict, str, int]]:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(GLINER).to("cuda")
    out = []
    for r in rows:
        names = {natural(t): t for t in r["requested_types"]}
        ents = model.predict_entities(r["text"], list(names), threshold=threshold)
        pred = {t: [] for t in r["requested_types"]}
        for e in ents:
            if e["text"] not in pred[names[e["label"]]]:
                pred[names[e["label"]]].append(e["text"])
        out.append((pred, json.dumps(ents, ensure_ascii=False), 0))
    return out


def parse_list(text: str) -> list[str] | None:
    start = text.find("[")
    if start < 0:
        return None
    obj = _loads_or_none(text[start:text.rfind("]") + 1]) if "]" in text else None
    if isinstance(obj, list):
        return [m for m in obj if isinstance(m, str)]
    for cand in iter_json_objects(text):  # a model that answers {"type": [...]} anyway
        o = _loads_or_none(cand)
        if isinstance(o, dict):
            return [m for v in o.values() for m in (v if isinstance(v, list) else [v]) if isinstance(m, str)]
    return None


def run_uniner(rows: list[dict]) -> list[tuple[dict, str, int]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=UNINER, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=2048)
    params = SamplingParams(temperature=0, max_tokens=256, stop=["</s>"])
    queries = [(i, t) for i, r in enumerate(rows) for t in r["requested_types"]]
    outs = llm.generate([UNINER_PROMPT.format(text=rows[i]["text"], etype=natural(t)) for i, t in queries], params)
    pred = [{t: [] for t in r["requested_types"]} for r in rows]
    raw = [{} for _ in rows]
    ntok = [0] * len(rows)
    for (i, t), o in zip(queries, outs):
        text = o.outputs[0].text
        raw[i][t] = text
        ntok[i] += len(o.outputs[0].token_ids)
        pred[i][t] = parse_list(text) or []
    return [(p, json.dumps(r, ensure_ascii=False), n) for p, r, n in zip(pred, raw, ntok)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gliner", "uniner"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--output-dir", default="outputs_tutor")
    ap.add_argument("--threshold", type=float, default=0.5, help="GLiNER only")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for f in args.data:
        name = Path(f).stem
        out = Path(args.output_dir) / name / args.run_name
        if (out / "metrics.json").exists():
            print(f"{args.run_name} {name}: already evaluated")
            continue
        rows = json.loads(Path(f).read_text(encoding="utf-8"))[: args.limit]
        t0 = time.time()
        results = run_gliner(rows, args.threshold) if args.model == "gliner" else run_uniner(rows)
        elapsed = time.time() - t0
        records = [{"id": r["id"], "requested_types": r["requested_types"], "distractors": r["distractors"],
                    "gold": r["gold"], "pred": p, "status": "ok", "finish_reason": "stop",
                    "reasoning_tokens": 0, "output_tokens": n, "raw": raw}
                   for r, (p, raw, n) in zip(rows, results)]
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        m = score(records) | {"run": args.run_name, "dataset": name, "model_id": GLINER if args.model == "gliner"
                              else UNINER, "threshold": args.threshold if args.model == "gliner" else None,
                              "generation_seconds": elapsed}
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"{args.run_name:<16} {name:<20} F1={m['F1']:.3f} (no MISC {m['without_MISC']['F1']:.3f}) "
              f"P={m['P']:.3f} R={m['R']:.3f} [{elapsed / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
