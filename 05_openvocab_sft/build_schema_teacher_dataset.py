"""Builds SFT datasets from the schema-guided teacher run (modal_teacher.py).

For each source example (same system + user prompt), the teacher's reasoning
and answer come from the same call, conditioned on the requested types. Two
variants:

- teacher: target = teacher reasoning + teacher answer (ReasoningNER-style:
  the labels are the teacher's own, consistent with its reasoning). Kept
  when the answer is a JSON object whose keys are exactly the requested
  types, with list values, and generation stopped normally.
- agree: only the examples where the teacher answer matches the Pile-NER
  gold on every requested type (spans compared on lower-cased word tokens);
  target = teacher reasoning + gold. Reasoning and gold are consistent by
  construction.

    python3 build_schema_teacher_dataset.py --source openvocab_train_v4_all.jsonl \\
        --teacher v4all_schema.jsonl --out-dir data/
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def norm(span: str) -> str:
    return " ".join(re.findall(r"\w+", span.lower()))


def parse_answer(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="open-vocabulary .jsonl the teacher was run on")
    ap.add_argument("--teacher", required=True, help="teacher/<tag>.jsonl from modal_teacher.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="openvocab_train_v5")
    args = ap.parse_args()

    src = {r["id"]: r for r in map(json.loads, Path(args.source).read_text(encoding="utf-8").splitlines()) if r}
    teacher = [json.loads(l) for l in Path(args.teacher).read_text(encoding="utf-8").splitlines() if l.strip()]
    stats, out_teacher, out_agree = Counter(), [], []
    for t in teacher:
        s = src.get(t["id"])
        if s is None:
            stats["missing_source"] += 1
            continue
        gold = json.loads(s["messages"][-1]["content"].rsplit("</think>", 1)[-1].strip())
        requested = list(gold)
        if t["finish_reason"] != "stop" or not t["reasoning"]:
            stats["truncated_or_no_reasoning"] += 1
            continue
        ans = parse_answer(t["answer"])
        if ans is None:
            stats["unparseable"] += 1
            continue
        if set(ans) != set(requested) or not all(isinstance(v, list) for v in ans.values()):
            stats["wrong_keys"] += 1
            continue
        ans = {k: list(dict.fromkeys(str(x) for x in ans[k])) for k in requested}
        base = [m for m in s["messages"][:-1]]
        out_teacher.append({"id": t["id"], "messages": base + [{"role": "assistant", "content":
                            f"<think>\n{t['reasoning']}\n</think>\n\n{json.dumps(ans, ensure_ascii=False)}"}]})
        agree = all({norm(x) for x in ans[k]} == {norm(x) for x in gold[k]} for k in requested)
        stats["agree" if agree else "disagree"] += 1
        if agree:
            out_agree.append({"id": t["id"], "messages": base + [{"role": "assistant", "content":
                              f"<think>\n{t['reasoning']}\n</think>\n\n{json.dumps(gold, ensure_ascii=False)}"}]})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("teacher", out_teacher), ("agree", out_agree)]:
        p = out / f"{args.prefix}_{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} examples -> {p}")
    print(dict(stats))


if __name__ == "__main__":
    main()
