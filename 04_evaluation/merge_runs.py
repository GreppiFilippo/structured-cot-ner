"""Joins the runs that together cover a set (e.g. a run on the first 100 rows
and one started at row 100 with eval_tutor.py --start 100, same settings) into
one run, and recomputes its metrics with eval_tutor.score.

Usage:
    python3 merge_runs.py --dir outputs_tutor --parts RUN_A RUN_B --out RUN --data tutor_eval/data/crossner_ai.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_tutor import score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="The sets, to check every query is covered once")
    args = ap.parse_args()
    for f in args.data:
        name = Path(f).stem
        ids = [r["id"] for r in json.loads(Path(f).read_text(encoding="utf-8"))]
        recs, metas = {}, []
        for part in args.parts:
            d = Path(args.dir) / name / part
            metas.append(json.loads((d / "metrics.json").read_text()))
            for line in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["id"] in recs:
                    raise SystemExit(f"{name}: {r['id']} is in more than one part")
                recs[r["id"]] = r
        missing = [i for i in ids if i not in recs]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} queries missing (e.g. {missing[0]})")
        keys = ("model_id", "adapter", "prompt", "grammar", "seed", "max_new_tokens", "sampling")
        for m in metas:  # runs made before --grammar existed decoded freely
            m.setdefault("grammar", "none")
        if any({k: m.get(k) for k in keys} != {k: metas[0].get(k) for k in keys} for m in metas):
            raise SystemExit(f"{name}: the parts were not run with the same settings")
        records = [recs[i] for i in ids]
        out = Path(args.dir) / name / args.out
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        m = score(records) | {k: metas[0].get(k) for k in keys} | {
            "run": args.out, "dataset": name, "merged_from": args.parts,
            "generation_seconds": sum(x.get("generation_seconds", 0) for x in metas)}
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"{args.out} {name}: {len(records)} queries, F1={m['F1']:.3f}, reasoning={m['mean_reasoning_tokens']:.0f} tok")


if __name__ == "__main__":
    main()
