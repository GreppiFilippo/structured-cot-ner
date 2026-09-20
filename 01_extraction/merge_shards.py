"""Merges all outputs/free_cot_*.jsonl shards into a single deduplicated file.

Run this on the node where the shards live (the master node, if you used the
-w faretra pin in sbatch_launch.sh) after all jobs have completed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--merged-path", type=str, default="outputs/free_cot_merged.jsonl")
    args = parser.parse_args()

    shard_paths = sorted(Path(args.output_dir).glob("free_cot_*_*.jsonl"))
    if not shard_paths:
        raise SystemExit(f"No shard files found in {args.output_dir}")

    seen_ids: set[str] = set()
    n_written = 0
    with open(args.merged_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as shard:
                for line in shard:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record["id"] in seen_ids:
                        continue  # can happen if a shard was re-run after a partial crash
                    seen_ids.add(record["id"])
                    out.write(line)
                    n_written += 1

    print(f"Merged {len(shard_paths)} shards into {args.merged_path}: {n_written} unique records.")


if __name__ == "__main__":
    main()
