"""Builds the two SFT files of the free vs structured CoT experiment from the
paired-trace dataset structured-cot-ner/pilener-teacher-cot-agreement-v1.

Each CoT is paired with the prompt it was generated from (free_cot with
free_prompt, structured_cot with structured_prompt); the target is the
teacher answer ("answer"), never the Pile-NER silver one. No system prompt.

The assistant turn is written as "<think>\\n{cot}</think>\\n\\n{answer}": with
the Qwen3-4B-Thinking-2507 chat template this renders exactly the layout of
the dataset card (checked on all 3,822 examples, both variants), and
train_qwen.py masks everything up to the generation prompt.

Usage:
    python3 build_paired_cot_dataset.py --out-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

REPO = "structured-cot-ner/pilener-teacher-cot-agreement-v1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = load_dataset(REPO, split="train")
    for variant in ("free", "structured"):
        path = out / f"pilener_paired_{variant}_cot.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in train:
                cot = r[f"{variant}_cot"]
                assert cot and "<think>" not in cot and "</think>" not in cot
                messages = [
                    {"role": "user", "content": r[f"{variant}_prompt"]},
                    {"role": "assistant", "content": f"<think>\n{cot}</think>\n\n{r['answer']}"},
                ]
                f.write(json.dumps({"id": r["example_id"], "messages": messages}, ensure_ascii=False) + "\n")
        print(f"{len(train)} examples -> {path}")


if __name__ == "__main__":
    main()
