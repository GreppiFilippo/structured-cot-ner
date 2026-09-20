"""Quick automated sanity checks over a smoke-test shard, before reading a
few examples by hand. This flags rows that deserve a closer look — it does
not replace manual review, the same way it hasn't anywhere else in this
project. Usage:

    python3 check_smoke_test.py outputs/free_cot_0_10.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def word_count(text: str) -> int:
    return len(text.split())


def has_repetition(text: str, min_len: int = 8, threshold: int = 3) -> bool:
    """Rough heuristic for the pathological looping behavior observed on
    some Qwen3.x variants earlier in this project (the same sentence-like
    chunk repeated many times): flags if any chunk of at least min_len
    words appears at least `threshold` times."""
    chunks = [c.strip() for c in text.replace("\n", " ").split(".") if len(c.split()) >= min_len]
    counts = Counter(chunks)
    return any(c >= threshold for c in counts.values())


def check_hf_backup(local_path: Path) -> None:
    """Best-effort check that the HF backup actually landed remotely, not
    just that push_now() didn't raise locally. Skipped silently if
    HF_BACKUP_REPO_ID / HF_TOKEN aren't set, or if huggingface_hub can't
    reach the network from here — this is a bonus check, not required for
    the rest of the script to work."""
    repo_id = os.environ.get("HF_BACKUP_REPO_ID", "")
    if not repo_id:
        print("\n(HF_BACKUP_REPO_ID not set in this shell — skipping remote backup check.)")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if local_path.name in files:
            print(f"\nHF backup confirmed: {local_path.name} is present in {repo_id}.")
        else:
            print(f"\nWARNING: {local_path.name} not found in {repo_id} — "
                  f"backup may not have fired yet, or push_every wasn't reached on only 10 rows "
                  f"(the unconditional end-of-run push should still have caught it — worth checking "
                  f"main.py's logs for 'Checkpoint push to HF Hub failed').")
    except Exception as error:
        print(f"\n(Could not verify remote backup from here: {error})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str, help="Path to the smoke-test .jsonl shard")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"{path} not found — has the smoke test job finished yet? Check with squeue.")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"{len(rows)} records in {path}\n")

    n_flagged = 0
    n_fallback = 0
    for row in rows:
        reasoning_words = word_count(row["reasoning"])
        thinking_words = word_count(row.get("native_thinking", ""))
        n_entities = len(row["entities"])
        n_excluded = len(row["excluded"])
        source = row.get("reasoning_source", "visible")  # older shards may not have this field

        flags = []
        if source == "native_thinking_fallback":
            flags.append("reasoning recovered from native_thinking (model skipped visible prose)")
            n_fallback += 1
        elif source == "none":
            flags.append("NO reasoning available at all (both visible and native_thinking empty)")
        if has_repetition(row.get("native_thinking", "")):
            flags.append("possible loop in native_thinking")
        if has_repetition(row["reasoning"]):
            flags.append("possible loop in reasoning")

        flag_str = f"  <-- {'; '.join(flags)}" if flags else ""
        if flags:
            n_flagged += 1
        print(f"{row['id']}: reasoning={reasoning_words}w ({source}), thinking={thinking_words}w, "
              f"entities={n_entities}, excluded={n_excluded}{flag_str}")

    print(f"\n{n_flagged}/{len(rows)} rows flagged for a closer look.")
    print(f"{n_fallback}/{len(rows)} rows needed the native_thinking fallback "
          f"(model skipped the visible prose reasoning despite the formatting rule). "
          f"Worth tracking this rate as you scale — it reflects a real, currently unresolved "
          f"model behavior, not a bug in the pipeline.")
    print("This is a quick automated scan (word counts + simple repetition "
          "check) — read at least 2-3 full examples by hand before trusting "
          "the pipeline on the full dataset, same as every prior test in "
          "this project.")

    check_hf_backup(path)


if __name__ == "__main__":
    main()
